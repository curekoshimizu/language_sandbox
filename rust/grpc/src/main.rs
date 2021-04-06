use tonic::{transport::Server, Request, Response, Status};

pub mod foo {
    tonic::include_proto!("foo");
}

use foo::foo_service_server::{FooService, FooServiceServer};
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "localhost:50051".parse().unwrap();

    let server = tokio::spawn(async move {
        let s = FooServer::default();
        Server::builder()
            .add_service(FooServiceServer::new(s))
            .serve(addr)
            .await
    });

    server.await??;

    Ok(())
}
