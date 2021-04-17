import add from './add_promise';

const hello = (name: string): string => `Hello, ${name}!`;

console.log(hello('World'));
console.log(add(2, 3));
