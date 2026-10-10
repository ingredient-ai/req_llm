defmodule ReqLLM.Streaming.ConnectionReuseTest do
  @moduledoc """
  A provider ends its stream with a terminal event (`data: [DONE]`) slightly
  before the HTTP response itself ends. The stream finishes on the terminal
  event, but the Finch request must still read the end of the response so
  Finch returns the connection to the pool. Killing the request at that point
  closes the connection, and every streamed call then opens a new TCP
  connection.
  """
  use ExUnit.Case, async: false

  alias ReqLLM.{Response, StreamResponse}

  # Time between the terminal SSE event and the end of the HTTP response.
  @response_tail_ms 100

  defmodule TailRouter do
    use Plug.Router

    plug(:match)
    plug(:dispatch)

    post "/v1/chat/completions" do
      conn =
        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_chunked(200)

      {:ok, conn} =
        Plug.Conn.chunk(
          conn,
          ~s(data: {"id":"r","object":"chat.completion.chunk","created":1,"model":"local","choices":[{"index":0,"delta":{"content":"Done"},"finish_reason":"stop"}]}\n\n)
        )

      {:ok, conn} = Plug.Conn.chunk(conn, "data: [DONE]\n\n")
      Process.sleep(Application.fetch_env!(:req_llm, :test_tail_ms))
      conn
    end
  end

  defmodule StallRouter do
    use Plug.Router

    plug(:match)
    plug(:dispatch)

    post "/v1/chat/completions" do
      conn =
        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_chunked(200)

      {:ok, conn} =
        Plug.Conn.chunk(
          conn,
          ~s(data: {"id":"r","object":"chat.completion.chunk","created":1,"model":"local","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n)
        )

      Process.sleep(10_000)
      conn
    end
  end

  setup do
    put_app_env(:test_tail_ms, @response_tail_ms)
  end

  test "a completed stream returns its connection to the pool for the next request" do
    {model, port} = start_server(TailRouter)
    # Always pick the first pool shard so both requests can share one connection.
    put_app_env(:stream_pool_strategy, &hd/1)
    attach_finch_events(port)

    assert "Done" = stream_text!(model)
    assert_receive {:finch, [:finch, :recv, :stop]}, 2_000

    assert "Done" = stream_text!(model)
    assert_receive {:finch, [:finch, :recv, :stop]}, 2_000

    assert_received {:finch, [:finch, :connect, :stop]}
    refute_received {:finch, [:finch, :connect, :stop]}
    assert_received {:finch, [:finch, :reused_connection]}
  end

  test "a response that never ends is abandoned after the drain timeout" do
    put_app_env(:test_tail_ms, 10_000)
    put_app_env(:stream_http_drain_timeout, 100)
    {model, _port} = start_server(TailRouter)

    assert "Done" = stream_text!(model)
    assert :ok = await_no_http_tasks(1_000)
  end

  test "a cancelled stream stops its request without waiting to drain" do
    put_app_env(:stream_http_drain_timeout, 10_000)
    {model, _port} = start_server(StallRouter)

    {:ok, stream_response} = ReqLLM.stream_text(model, "Say", api_key: "test-key", max_retries: 0)
    assert [_chunk] = Enum.take(stream_response.stream, 1)

    assert :ok = await_no_http_tasks(1_000)
  end

  defp put_app_env(key, value) do
    previous = Application.fetch_env(:req_llm, key)
    Application.put_env(:req_llm, key, value)

    on_exit(fn ->
      case previous do
        {:ok, previous_value} -> Application.put_env(:req_llm, key, previous_value)
        :error -> Application.delete_env(:req_llm, key)
      end
    end)
  end

  defp start_server(router) do
    server = start_supervised!({Bandit, plug: router, port: 0, ip: {127, 0, 0, 1}})
    {:ok, {_address, port}} = ThousandIsland.listener_info(server)

    model =
      ReqLLM.model!(%{
        id: "local-chat",
        provider: :openai,
        base_url: "http://127.0.0.1:#{port}/v1",
        extra: %{wire: %{protocol: "openai_chat"}}
      })

    {model, port}
  end

  defp stream_text!(model) do
    {:ok, stream_response} = ReqLLM.stream_text(model, "Say", api_key: "test-key", max_retries: 0)

    {:ok, response} = StreamResponse.to_response(stream_response)
    Response.text(response)
  end

  defp attach_finch_events(port) do
    test_pid = self()
    handler_id = {__MODULE__, make_ref()}

    :telemetry.attach_many(
      handler_id,
      [[:finch, :connect, :stop], [:finch, :reused_connection], [:finch, :recv, :stop]],
      fn event, _measurements, metadata, _config ->
        if request_port(metadata) == port, do: send(test_pid, {:finch, event})
      end,
      nil
    )

    on_exit(fn -> :telemetry.detach(handler_id) end)
  end

  defp request_port(%{port: port}), do: port
  defp request_port(%{request: %{port: port}}), do: port
  defp request_port(_metadata), do: nil

  defp await_no_http_tasks(timeout) when timeout <= 0, do: {:error, :http_task_alive}

  defp await_no_http_tasks(timeout) do
    case Task.Supervisor.children(ReqLLM.TaskSupervisor) do
      [] ->
        :ok

      _children ->
        Process.sleep(20)
        await_no_http_tasks(timeout - 20)
    end
  end
end
