pub trait HasName {
    fn name(&self) -> String;
}

#[derive(Debug)]
pub struct Dog {}

impl Dog {
    pub fn new() -> Self {
        Dog {}
    }
    pub fn bark(&self) -> String {
        "woof!".to_string()
    }
}

impl HasName for Dog {
    fn name(&self) -> String {
        "dog".to_string()
    }
}

pub struct Adapter<T, F> {
    _obj: T,
    _make_noise: F,
}

impl<T: HasName, F: Fn(&T) -> String> Adapter<T, F> {
    pub fn new(obj: T, noise: F) -> Self {
        Adapter {
            _obj: obj,
            _make_noise: noise,
        }
    }

    pub fn make_noise(&self) -> String {
        (self._make_noise)(&self._obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter() {
        let dog = Dog::new();
        println!("{:?}", dog);

        let adapter = Adapter::new(dog, |x| x.bark());
        adapter.make_noise();
    }
}
