from marshmallow import Schema, fields, validate, validates_schema

class ConfigSchema(Schema):

    class _AreaSchema(Schema):
        x = fields.Integer(required=True)
        y = fields.Integer(required=True)

    version = fields.Str()
    name = fields.Str(required=True)
    workarea = fields.Nested(_AreaSchema, required=True)
    qsamplemin = fields.Integer(required=True)
    qsamplemax = fields.Integer(required=True)
    samlesizemax = fields.Integer(required=True)
    size = fields.Integer(required=True)
    crossover = fields.Float(required=True)
    mutation = fields.Float(required=True)
    iterations = fields.Integer(required=True)
    fittestAlwaysSurvives = fields.Boolean(required=True)
    maxResults = fields.Integer(required=True)
    skip = fields.Integer(required=True)
    drift = fields.Float(required=True)
    debug = fields.Boolean(required=True)
