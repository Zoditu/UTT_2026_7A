const Joi = require('joi')
const app = require('express');

const schema_usuario = Joi.object({
    username: Joi.string().alphanum().min(3).max(30).required(),
    email: Joi.string().email().required(),
    age: Joi.number().integer().min(18),
    x: Joi.string().valid("Ejemplo", "Papa")
});

const { error, value } = schema_usuario.validate({
    username: 'danilo',
    email: 'danilo@example.com',
    age: 28,
    x: "Z"
});

console.log(`Error: ${JSON.stringify(error)}`);
console.log(`Resultado: ${JSON.stringify(value)}`);