use std::sync::Mutex;

use axum::response::Html;
use axum::Json;
use axum::{routing::get, Router};
use serde::Deserialize;

#[tokio::main]
async fn main() {
    // TODO: logger
    // TODO: not found
    let app = Router::new()
        .route("/users", get(handler))
        .route("/", get(root));

    //info!("hello");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello world!"
}

#[derive(Deserialize)]
struct RangeParameters {
    start: usize,
    end: usize,
}

async fn handler(Json(payload): Json<RangeParameters>) -> Html<String> {
    println!("{0:?}~{1:?} : dice", payload.start, payload.end);

    Html(format!("<h1>Hi</h1>"))
}
