use tokio::net::TcpListener;

use axum::{response::IntoResponse, routing, Json, Router};
use serde::{Deserialize, Serialize};

use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

#[derive(Serialize, Deserialize, Debug, ToSchema)]
struct RangeParameters {
    start: usize,
    end: usize,
}

#[derive(OpenApi)]
#[openapi(paths(handler), components(schemas(RangeParameters)))]
struct ApiDoc;

fn create_router() -> Router {
    let app = Router::new()
        .route("/users", routing::get(handler))
        .route("/", routing::get(root));

    app.merge(SwaggerUi::new("/swagger-ui").url("/api-doc/openapi.json", ApiDoc::openapi()))
}

#[tokio::main]
async fn main() {
    // TODO: logger
    // TODO: not found

    let app = create_router();
    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello world!"
}

#[utoipa::path(
    get,
    path = "/users/",
    responses(
        (status = 200, body = [RangeParameters])
    ),
)]
async fn handler(Json(payload): Json<RangeParameters>) -> impl IntoResponse {
    println!("{0:?}~{1:?} : dice", payload.start, payload.end);

    Json(vec![payload])
}
