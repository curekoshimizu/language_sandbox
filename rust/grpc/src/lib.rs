use std::io;
use std::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

pub mod foo {
    tonic::include_proto!("foo");
}

use foo::foo_service_server::FooService;
use foo::{RequestMsg, ResponseMsg};

#[derive(Default)]
pub struct FooServer {}

#[tonic::async_trait]
impl FooService for FooServer {
    async fn foo(&self, request: Request<RequestMsg>) -> Result<Response<ResponseMsg>, Status> {
        let reply = ResponseMsg {
            response: format!("{} hello", request.get_ref().prefix),
        };

        Ok(Response::new(reply))
    }

    type FooStreamStream = ReceiverStream<Result<ResponseMsg, Status>>;

    async fn foo_stream(
        &self,
        request: Request<RequestMsg>,
    ) -> Result<Response<Self::FooStreamStream>, Status> {
        let (tx, rx) = mpsc::channel(2);

        let recv_stream = ReceiverStream::new(rx);

        let reply = ResponseMsg {
            response: format!("{} hello", request.get_ref().prefix),
        };
        tx.send(Ok(reply.clone())).await.unwrap();
        tx.send(Ok(reply)).await.unwrap();

        Ok(Response::new(recv_stream))
    }
}

pub fn available_port() -> io::Result<u16> {
    match TcpListener::bind("localhost:0") {
        Ok(listener) => Ok(listener.local_addr().unwrap().port()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use foo::foo_service_client::FooServiceClient;
    use foo::foo_service_server::FooServiceServer;
    use futures_util::FutureExt;
    use std::net::SocketAddr;
    use tokio::sync::oneshot;
    use tonic::transport::Server;

    #[tokio::test]
    async fn grpc_foo_test() -> Result<(), Box<dyn std::error::Error>> {
        let port = available_port()?;
        let address = format!("127.0.0.1:{}", port);
        let socket_addr = address.parse::<SocketAddr>().unwrap();

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let s = FooServer::default();
        let server = Server::builder().add_service(FooServiceServer::new(s));
        let (ready_tx, ready_rx) = oneshot::channel();

        let server_future = tokio::spawn(async move {
            ready_tx.send(()).unwrap(); // I am ready.

            server
                .serve_with_shutdown(socket_addr, shutdown_rx.map(|rx| drop(rx)))
                .await
        });

        let received = ready_rx.await?;
        assert_eq!(received, ());

        {
            let mut client = FooServiceClient::connect(format!("http://{}", address)).await?;

            let request = tonic::Request::new(RequestMsg {
                prefix: "`from client`".to_string(),
            });

            let response = client.foo(request).await?;

            assert_eq!(response.get_ref().response, "`from client` hello");
        }

        shutdown_tx.send(()).unwrap();

        server_future.await??;

        Ok(())
    }
}
