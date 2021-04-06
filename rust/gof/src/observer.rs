pub trait ObserverInterface {
    fn on_notify(&self, message: &str);
}

pub struct Observer {
    prefix: String,
}

impl Observer {
    pub fn new(prefix: String) -> Observer {
        Observer { prefix: prefix }
    }
}

impl ObserverInterface for Observer {
    fn on_notify(&self, message: &str) {
        println!("{} received {:?}", self.prefix, message);
    }
}

pub trait SubjectInterface<'a> {
    fn notify(&self, message: &str) -> usize;
    fn subscribe_ref(&mut self, observer: &'a dyn ObserverInterface);
    fn unsubscribe_ref(&mut self, observer: &'a dyn ObserverInterface);
    fn subscribe_box(&mut self, observer: &'a Box<dyn ObserverInterface>);
    fn unsubscribe_box(&mut self, observer: &'a Box<dyn ObserverInterface>);
}

pub struct Subject<'a> {
    observers_ref: Vec<&'a dyn ObserverInterface>,
    observers_box: Vec<&'a Box<dyn ObserverInterface>>,
}

impl<'a> Subject<'a> {
    pub fn new() -> Subject<'a> {
        Subject {
            observers_ref: Vec::new(),
            observers_box: Vec::new(),
        }
    }
}

impl<'a> SubjectInterface<'a> for Subject<'a> {
    fn notify(&self, message: &str) -> usize {
        let mut count = 0;
        for observer in self.observers_ref.iter() {
            observer.on_notify(message);
            count += 1;
        }
        for observer in self.observers_box.iter() {
            observer.on_notify(message);
            count += 1;
        }
        count
    }

    fn subscribe_ref(&mut self, observer: &'a dyn ObserverInterface) {
        self.observers_ref.push(observer);
    }

    fn unsubscribe_ref(&mut self, observer: &'a dyn ObserverInterface) {
        self.observers_ref
            .retain(|&obs| !std::ptr::eq(obs, observer));
    }

    fn subscribe_box(&mut self, observer: &'a Box<dyn ObserverInterface>) {
        self.observers_box.push(observer);
    }

    fn unsubscribe_box(&mut self, observer: &'a Box<dyn ObserverInterface>) {
        self.observers_box
            .retain(|&obs| !std::ptr::eq(obs, observer));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer() {
        let mut subject = Subject::new();

        let observer_a = Observer::new("Observer A.".to_string());
        let observer_b: Box<dyn ObserverInterface> =
            Box::new(Observer::new("Observer B.".to_string()));

        subject.subscribe_ref(&observer_a);
        subject.subscribe_box(&observer_b);

        subject.notify("Hello Observers");

        subject.unsubscribe_box(&observer_b);
        subject.notify("Hello Observers!");
        subject.unsubscribe_ref(&observer_a);
        subject.notify("Hello Observers!!");
    }
}
