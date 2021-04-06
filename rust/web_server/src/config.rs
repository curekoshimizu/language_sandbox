use actix_web::{get, web, Responder};
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

async fn counter_resp(data: web::Data<AppStateWithCounter>) -> String {
    let mut counter = data.counter.lock().unwrap();
    *counter += 1;

    format!("Request number: {}", counter)
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
