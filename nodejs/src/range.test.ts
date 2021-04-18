import range, { counter } from './range';

describe('range test', () => {
  test('range', () => {
    const array = range(5);
    expect(array.length).toBe(5);

    const squared = array.map((x) => x * x);

    squared.forEach((x, i) => {
      expect(i * i).toBe(x);
    });
  });

  test('count', () => {
    const gen = counter();
    expect(gen.next().value).toBe(0);
    expect(gen.next().value).toBe(1);
    expect(gen.next().value).toBe(2);
    expect(gen.next().value).toBe(3);

    let sum = 0;
    for (const x of counter()) {
      if (x > 3) {
        break;
      }
      sum += x;
    }
    expect(sum).toBe(6);
  });
});
