const add = (x: number, y: number) : number => x + y;
export default add;

export const addWithCallback = (x: number, y: number, callback: (arg: number) => void) : void => {
  setTimeout(() => callback(x + y), 100);
};

export const addPromise = (x: number, y: number): Promise<number> => new Promise((resolve) => setTimeout(() => resolve(x + y), 100));

export const addPromiseFromCallback = (x: number, y: number): Promise<number> => new Promise((resolve) => addWithCallback(x, y, resolve));
