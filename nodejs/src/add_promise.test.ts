import add, { addWithCallback, addPromise, addPromiseFromCallback } from './add_promise';

const ans = 5;

describe('promise / async test', () => {
  test('add', () => {
    expect(add(2, 3)).toBe(ans);
  });

  test('add callback', (done) => {
    const verify = (ret: number):void => {
      expect(ret).toBe(ans);
      done();
    };

    addWithCallback(2, 3, verify);
  });

  test('add promise', () => {
    const verify = (ret: number):void => {
      expect(ret).toBe(ans);
    };

    return addPromise(2, 3).then(verify);
  });

  test('add promise 2', () => {
    const verify = (ret: number):void => {
      expect(ret).toBe(ans);
    };

    return addPromiseFromCallback(2, 3).then(verify);
  });

  test('add async', async () => {
    const ret = await addPromise(2, 3);
    expect(ret).toBe(ans);
  });

  test('add async 2', async () => {
    const ret = await addPromiseFromCallback(2, 3);
    expect(ret).toBe(ans);
  });
});
