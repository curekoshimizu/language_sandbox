use std::marker::PhantomData;

pub trait Animal {
    fn new() -> Self;
    fn speak(&self) -> &str;
}

struct Dog;
struct Cat;

impl Animal for Dog {
    fn new() -> Self {
        Dog {}
    }
    fn speak(&self) -> &str {
        "woof"
    }
}

impl Animal for Cat {
    fn new() -> Self {
        Cat {}
    }
    fn speak(&self) -> &str {
        "meow"
    }
}

pub struct PetShop<T: Animal> {
    _phantom: PhantomData<T>,
}

impl<T: Animal> PetShop<T> {
    pub fn new() -> Self {
        PetShop {
            _phantom: PhantomData,
        }
    }
    pub fn show_pet(&self) -> String {
        let pet = T::new();
        pet.speak().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abstract_factory() {
        let cat_shop: PetShop<Cat> = PetShop::new();
        println!("{}", cat_shop.show_pet());

        let dog_shop: PetShop<Dog> = PetShop::new();
        println!("{}", dog_shop.show_pet());
    }
}
