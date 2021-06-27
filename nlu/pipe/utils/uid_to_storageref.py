"""Helper mappings to resolve storage ref for some healthcare models.
Sometimes Storageref==UID on some Annotators. To resolve to correct embedding provider, we must define a mapping from UID->storageref
"""
mappings = {
'RelationExtractionModel_9c255241fec3':'clinical',
'RelationExtractionModel_14b00157fc1a' : 'clinical',

}
