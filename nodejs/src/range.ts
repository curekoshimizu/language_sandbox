const range = (n:number):number[] => [...Array(n).keys()];
export default range;

/* eslint func-style: off */
export function* counter():Generator<number> {
  let index = 0;

  while (true) {
    yield index;
    index += 1;
  }
}
