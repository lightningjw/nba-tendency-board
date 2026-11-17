// Minimal JSON Schema validator for the contract schemas.
// Supports exactly the keywords the schemas are allowed to use (see tendency/contract.py
// SUPPORTED_KEYWORDS, enforced by tests/test_models_contract.py): type, properties, required,
// additionalProperties, items, enum, const, minimum, maximum, minItems, maxItems, $ref, $defs.

function typeOf(v) {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  if (typeof v === "number") return Number.isInteger(v) ? "integer" : "number";
  return typeof v;
}

function typeMatches(v, t) {
  const actual = typeOf(v);
  return actual === t || (t === "number" && actual === "integer");
}

export function validate(value, schema, root = schema, path = "", errors = []) {
  if (errors.length >= 8) return errors;
  if (schema.$ref) {
    const name = schema.$ref.replace("#/$defs/", "");
    const target = root.$defs && root.$defs[name];
    if (!target) errors.push(`${path || "/"}: unknown $ref ${schema.$ref}`);
    else validate(value, target, root, path, errors);
    return errors;
  }
  if ("const" in schema && value !== schema.const) errors.push(`${path || "/"}: expected ${JSON.stringify(schema.const)}`);
  if (schema.enum && !schema.enum.includes(value)) errors.push(`${path || "/"}: ${JSON.stringify(value)} is not one of ${JSON.stringify(schema.enum)}`);
  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => typeMatches(value, t))) {
      errors.push(`${path || "/"}: expected ${types.join(" or ")}, got ${typeOf(value)}`);
      return errors;
    }
  }
  if (typeof value === "number") {
    if ("minimum" in schema && value < schema.minimum) errors.push(`${path}: ${value} is below ${schema.minimum}`);
    if ("maximum" in schema && value > schema.maximum) errors.push(`${path}: ${value} is above ${schema.maximum}`);
  }
  if (Array.isArray(value)) {
    if ("minItems" in schema && value.length < schema.minItems) errors.push(`${path}: needs at least ${schema.minItems} items`);
    if ("maxItems" in schema && value.length > schema.maxItems) errors.push(`${path}: allows at most ${schema.maxItems} items`);
    if (schema.items) for (let i = 0; i < value.length && errors.length < 8; i++) validate(value[i], schema.items, root, `${path}/${i}`, errors);
  } else if (value !== null && typeof value === "object") {
    const props = schema.properties || {};
    for (const key of schema.required || []) if (!(key in value)) errors.push(`${path || "/"}: missing "${key}"`);
    for (const key of Object.keys(value)) {
      if (key in props) validate(value[key], props[key], root, `${path}/${key}`, errors);
      else if (schema.additionalProperties === false) errors.push(`${path || "/"}: unexpected "${key}"`);
    }
  }
  return errors;
}
