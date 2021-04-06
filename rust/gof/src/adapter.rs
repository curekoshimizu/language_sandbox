pub trait HasName {
    fn name(&self) -> String;
}

#[derive(Debug)]
pub struct Dog {}
pub struct Cat {}

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

impl Cat {
    pub fn new() -> Self {
        Cat {}
    }
    pub fn meow(&self) -> String {
        "meow!".to_string()
    }
}

impl HasName for Cat {
    fn name(&self) -> String {
        "cat".to_string()
    }
}

// pub struct Adapter<T, F> {
//     _obj: T,
//     _make_noise: F,
// }
//
// impl<T: HasName, F: Fn(&T) -> String> Adapter<T, F> {
//     pub fn new(obj: T, noise: F) -> Self {
//         Adapter {
//             _obj: obj,
//             _make_noise: noise,
//         }
//     }
//
//     pub fn make_noise(&self) -> String {
//         (self._make_noise)(&self._obj)
//     }
// }

pub trait Adapter {
    fn make_noise(&self) -> String;
}

impl Adapter for Dog {
    fn make_noise(&self) -> String {
        self.bark()
    }
}
impl Adapter for Cat {
    fn make_noise(&self) -> String {
        self.meow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter() {
        let mut objects: Vec<Box<dyn Adapter>> = Vec::new();
        objects.push(Box::new(Dog::new()));
        objects.push(Box::new(Cat::new()));

        for obj in objects.iter() {
            // println!("{} says {}", obj.name(), obj.make_noise());
            println!("it says {}", obj.make_noise());
        }
    }
}
