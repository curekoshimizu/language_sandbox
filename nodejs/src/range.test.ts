import range from './range';

test('range', () => {
  const array = range(5);
  expect(array.length).toBe(5);

  const squared = array.map((x) => x * x);

  squared.forEach((x, i) => {
    expect(i * i).toBe(x);
  });
});
