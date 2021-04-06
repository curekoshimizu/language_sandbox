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
