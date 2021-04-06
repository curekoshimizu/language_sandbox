pub enum ChairType {
    BigChair,
    MediumChair,
    SmallChair,
}

pub struct Chair {
    t: ChairType,
}

impl Chair {
    pub fn new(t: ChairType) -> Chair {
        Chair { t: t }
    }

    pub fn dimension(&self) -> (usize, usize, usize) {
        match self.t {
            ChairType::BigChair => (80, 80, 80),
            ChairType::MediumChair => (80, 60, 60),
            ChairType::SmallChair => (40, 40, 80),
        }
    }
}

impl ChairType {
    pub fn dimension(&self) -> (usize, usize, usize) {
        match self {
            ChairType::BigChair => (80, 80, 80),
            ChairType::MediumChair => (80, 60, 60),
            ChairType::SmallChair => (40, 40, 80),
        }
    }
}

pub struct ChairFactory {}

impl ChairFactory {
    pub fn get_chair(c: ChairType) -> Chair {
        Chair::new(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory() {
        let big_factory = ChairFactory::get_chair(ChairType::BigChair);
        assert_eq!(big_factory.dimension(), (80, 80, 80));
        let small_factory = ChairFactory::get_chair(ChairType::SmallChair);
        assert_eq!(small_factory.dimension(), (40, 40, 80));
    }
}
