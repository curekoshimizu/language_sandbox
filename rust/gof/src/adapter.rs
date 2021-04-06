pub trait HasName {
    fn name(&self) -> String;
}

#[derive(Debug)]
pub struct Dog {}
pub struct Cat {}
pub struct Human {}
pub struct Car {}

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

impl Human {
    pub fn new() -> Self {
        Human {}
    }
    pub fn hello(&self) -> String {
        "hello!".to_string()
    }
}

impl HasName for Human {
    fn name(&self) -> String {
        "human".to_string()
    }
}

impl Car {
    pub fn new() -> Self {
        Car {}
    }
    pub fn noise(&self, level: usize) -> String {
        "vroom ".repeat(level).to_string() + "!"
    }
}

impl HasName for Car {
    fn name(&self) -> String {
        "car".to_string()
    }
}

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
impl Adapter for Human {
    fn make_noise(&self) -> String {
        self.hello()
    }
}
impl Adapter for Car {
    fn make_noise(&self) -> String {
        self.noise(2)
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
        objects.push(Box::new(Human::new()));
        objects.push(Box::new(Car::new()));

        for obj in objects.iter() {
            // println!("{} says {}", obj.name(), obj.make_noise());
            println!("it says {}", obj.make_noise());
        }
    }
}
