use actix_web::{App, HttpServer};
use env_logger;
use std::env;

#[macro_use]
extern crate log;

mod config;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let server = HttpServer::new(|| App::new().configure(config::config)).bind("127.0.0.1:8080")?;
    info!("server is running");
    server.run().await
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_hoge() -> Result<(), reqwest::Error> {
        use serde_json::{json, Value};

        use httpmock::Method::POST;
        use httpmock::MockServer;

        // Arrange
        let server = MockServer::start_async().await;

        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/users")
                .header("Content-Type", "application/json")
                .json_body(json!({ "name": "hoge" }));
            then.status(201)
                .header("Content-Type", "application/json")
                .json_body(json!({ "id": 1234 }));
        });

        let client = reqwest::Client::new();

        let resp = client
            .post(&format!("http://{}/users", server.address()))
            .json(&json![{"name": "hoge"}])
            .send()
            .await?;

        assert_eq!(resp.status(), 201);
        let ret: Value = resp.json().await?;
        assert_eq!(ret, json!({ "id": 1234} ));

        m.assert();

        Ok(())
    }
}
