import { EdgeTypes } from 'reactflow';
import RelationshipEdge from './RelationshipEdge';

export const edgeTypes: EdgeTypes = {
  relationship: RelationshipEdge
};

export { default as RelationshipEdge } from './RelationshipEdge';
