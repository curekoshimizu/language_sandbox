use actix_web::{get, web, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

// This struct represents state
struct AppState {
    app_name: String,
}

#[get("/")]
async fn app_name_resp(data: web::Data<AppState>) -> String {
    let app_name = &data.app_name; // <- get app_name

    format!("Hello {}!", app_name) // <- response with app_name
}

struct AppStateWithCounter {
    counter: Mutex<i32>, // <- Mutex is necessary to mutate safely across threads
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq)]
pub struct CounterResponse {
    counter: i32,
    id: i32,
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq)]
pub struct UserData {
    id: i32,
}

async fn counter_resp(
    data: web::Data<AppStateWithCounter>,
    user_data: web::Json<UserData>,
) -> web::Json<CounterResponse> {
    let mut counter = data.counter.lock().unwrap();
    *counter += 1;

    web::Json(CounterResponse {
        counter: *counter,
        id: user_data.id,
    })
}

async fn hello() -> impl Responder {
    "Hello world!"
}

pub fn config(cfg: &mut web::ServiceConfig) {
    let counter = web::Data::new(AppStateWithCounter {
        counter: Mutex::new(0),
    });

    cfg //
        .data(AppState {
            app_name: String::from("Actix-web"),
        })
        .service(app_name_resp)
        .app_data(counter.clone())
        .route("/counter", web::get().to(counter_resp))
        .service(web::scope("/app").route("/index.html", web::get().to(hello)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{http, test, App};

    #[actix_rt::test]
    async fn async_test() {
        let mut app = test::init_service(App::new().configure(config)).await;
        let req = test::TestRequest::with_header("content-type", "text/plain").to_request();
        let resp = test::call_service(&mut app, req).await;
        assert_eq!(resp.status(), http::StatusCode::OK);
    }

    #[actix_rt::test]
    async fn test_root_response() {
        let mut app = test::init_service(App::new().configure(config)).await;

        {
            // GET
            let req = test::TestRequest::get().uri("/").to_request();
            let resp = test::call_service(&mut app, req).await;
            assert_eq!(resp.status().as_u16(), 200);

            let bytes = test::read_body(resp).await;
            let body = String::from_utf8(bytes.to_vec()).unwrap();
            assert_eq!(body, "Hello Actix-web!");
        }

        {
            // POST
            let req = test::TestRequest::post().uri("/").to_request();
            let resp = test::call_service(&mut app, req).await;
            assert_eq!(resp.status().as_u16(), 404);

            let bytes = test::read_body(resp).await;
            assert_eq!(bytes.len(), 0);
        }
    }

    #[actix_rt::test]
    async fn test_counter() {
        let mut app = test::init_service(App::new().configure(config)).await;

        {
            // 1st try
            let req = test::TestRequest::get()
                .uri("/counter")
                .set_json(&UserData { id: 123 })
                .to_request();
            let resp = test::call_service(&mut app, req).await;
            assert_eq!(resp.status(), http::StatusCode::OK);

            let resp_json: CounterResponse = test::read_body_json(resp).await;
            assert_eq!(
                resp_json,
                CounterResponse {
                    counter: 1,
                    id: 123
                }
            );
        }

        {
            // 2nd try
            let req = test::TestRequest::get()
                .uri("/counter")
                .set_json(&UserData { id: 0x123 })
                .to_request();
            let resp = test::call_service(&mut app, req).await;
            assert_eq!(resp.status(), http::StatusCode::OK);

            let resp_json: CounterResponse = test::read_body_json(resp).await;
            assert_eq!(
                resp_json,
                CounterResponse {
                    counter: 2,
                    id: 0x123
                }
            );
        }

        {
            // no param
            let req = test::TestRequest::get().uri("/counter").to_request();
            let resp = test::call_service(&mut app, req).await;
            assert_eq!(resp.status(), http::StatusCode::BAD_REQUEST);
        }
    }
}
