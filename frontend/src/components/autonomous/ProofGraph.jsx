import React, { useMemo, useState } from 'react';
import './ProofGraph.css';

const NODE_WIDTH = 236;
const NODE_HEIGHT = 110;
const X_GAP = 72;
const Y_GAP = 30;
const PADDING = 28;

function truncate(text, maxLength = 96) {
  if (!text) {
    return '';
  }
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
}

function wrapSvgText(text, maxCharsPerLine = 30, maxLines = 3) {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim();
  if (!normalized) {
    return ['Untitled proof'];
  }

  const words = normalized.split(' ');
  const lines = [];
  let currentLine = '';

  for (const word of words) {
    const candidate = currentLine ? `${currentLine} ${word}` : word;
    if (candidate.length <= maxCharsPerLine) {
      currentLine = candidate;
      continue;
    }

    if (currentLine) {
      lines.push(currentLine);
    }
    currentLine = word;

    if (lines.length === maxLines - 1) {
      break;
    }
  }

  if (currentLine && lines.length < maxLines) {
    lines.push(currentLine);
  }

  if (lines.length === maxLines && words.join(' ').length > lines.join(' ').length) {
    const lastLine = lines[maxLines - 1];
    lines[maxLines - 1] = truncate(lastLine, Math.max(8, maxCharsPerLine - 3));
  }

  return lines;
}

function ProofGraph({
  nodes = [],
  edgesMoto = [],
  edgesMathlib = [],
  visibleProofIds = [],
  expandedProofId = null,
  onSelectProof,
}) {
  const [hoveredProofId, setHoveredProofId] = useState(null);

  const visibleIdSet = useMemo(() => new Set(visibleProofIds), [visibleProofIds]);

  const graphData = useMemo(() => {
    const filteredNodes = nodes.filter((node) => visibleIdSet.has(node.proof_id));
    const nodeById = new Map(filteredNodes.map((node) => [node.proof_id, node]));

    const filteredMotoEdges = edgesMoto.filter(
      (edge) => nodeById.has(edge.from) && nodeById.has(edge.to)
    );

    const mathlibByProofId = new Map();
    for (const edge of edgesMathlib) {
      if (!nodeById.has(edge.from)) {
        continue;
      }
      const entries = mathlibByProofId.get(edge.from) || [];
      entries.push(edge);
      mathlibByProofId.set(edge.from, entries);
    }

    const dependenciesByProofId = new Map();
    for (const node of filteredNodes) {
      dependenciesByProofId.set(node.proof_id, []);
    }
    for (const edge of filteredMotoEdges) {
      dependenciesByProofId.get(edge.from).push(edge.to);
    }

    const layerCache = new Map();
    const visiting = new Set();
    const getLayer = (proofId) => {
      if (layerCache.has(proofId)) {
        return layerCache.get(proofId);
      }
      if (visiting.has(proofId)) {
        return 0;
      }

      visiting.add(proofId);
      const dependencies = dependenciesByProofId.get(proofId) || [];
      const layer = dependencies.length > 0
        ? 1 + Math.max(...dependencies.map((dependencyId) => getLayer(dependencyId)))
        : 0;
      visiting.delete(proofId);
      layerCache.set(proofId, layer);
      return layer;
    };

    const layers = new Map();
    for (const node of filteredNodes) {
      const layer = getLayer(node.proof_id);
      const entries = layers.get(layer) || [];
      entries.push(node);
      layers.set(layer, entries);
    }

    const orderedLayers = Array.from(layers.entries())
      .sort((left, right) => left[0] - right[0])
      .map(([layer, layerNodes]) => {
        const sortedNodes = [...layerNodes].sort((left, right) => {
          const leftName = left.theorem_name || left.theorem_statement || left.proof_id;
          const rightName = right.theorem_name || right.theorem_statement || right.proof_id;
          return leftName.localeCompare(rightName);
        });
        return { layer, nodes: sortedNodes };
      });

    const maxLayerHeight = Math.max(
      ...orderedLayers.map(({ nodes: layerNodes }) => (
        layerNodes.length * NODE_HEIGHT + Math.max(layerNodes.length - 1, 0) * Y_GAP
      )),
      NODE_HEIGHT
    );

    const width = Math.max(
      PADDING * 2 + NODE_WIDTH,
      PADDING * 2 + orderedLayers.length * NODE_WIDTH + Math.max(orderedLayers.length - 1, 0) * X_GAP
    );
    const height = PADDING * 2 + maxLayerHeight;

    const positions = new Map();
    orderedLayers.forEach(({ layer, nodes: layerNodes }) => {
      const layerHeight = layerNodes.length * NODE_HEIGHT + Math.max(layerNodes.length - 1, 0) * Y_GAP;
      const startY = PADDING + Math.max((maxLayerHeight - layerHeight) / 2, 0);
      const x = PADDING + layer * (NODE_WIDTH + X_GAP);

      layerNodes.forEach((node, index) => {
        positions.set(node.proof_id, {
          x,
          y: startY + index * (NODE_HEIGHT + Y_GAP),
        });
      });
    });

    return {
      filteredNodes,
      filteredMotoEdges,
      mathlibByProofId,
      nodeById,
      positions,
      width,
      height,
    };
  }, [nodes, edgesMathlib, edgesMoto, visibleIdSet]);

  const hoveredNode = hoveredProofId ? graphData.nodeById.get(hoveredProofId) : null;
  const hoveredMathlibRefs = hoveredProofId
    ? (graphData.mathlibByProofId.get(hoveredProofId) || [])
    : [];

  if (graphData.filteredNodes.length === 0) {
    return (
      <div className="proof-graph-empty">
        No proofs match the current filter. Switch to "All Verified Proofs" to inspect the full graph.
      </div>
    );
  }

  return (
    <div className="proof-graph-panel">
      <div className="proof-graph-topbar">
        <div className="proof-graph-copy">
          <strong>Proof ancestry graph</strong>
          <span>
            Showing {graphData.filteredNodes.length} proof{graphData.filteredNodes.length === 1 ? '' : 's'}.
            Solid arrows run from a dependency to the proof that uses it.
          </span>
        </div>

        <div className="proof-graph-hover-card">
          {hoveredNode ? (
            <>
              <div className="proof-graph-hover-title">
                {hoveredNode.theorem_name || hoveredNode.proof_id}
              </div>
              <div className="proof-graph-hover-subtitle">
                {truncate(hoveredNode.theorem_statement, 150)}
              </div>
              <div className="proof-graph-badges">
                {hoveredMathlibRefs.length > 0 ? (
                  hoveredMathlibRefs.slice(0, 8).map((reference, index) => (
                    <span
                      key={`${reference.from}-${reference.name}-${index}`}
                      className="proof-graph-badge"
                      title={reference.source_ref || reference.name}
                    >
                      {reference.name}
                    </span>
                  ))
                ) : (
                  <span className="proof-graph-badge muted">No Mathlib references tracked</span>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="proof-graph-hover-title">Hover a node</div>
              <div className="proof-graph-hover-subtitle">
                Hover a proof node to inspect its tracked Mathlib references.
              </div>
            </>
          )}
        </div>
      </div>

      <div className="proof-graph-canvas">
        <svg
          className="proof-graph-svg"
          viewBox={`0 0 ${graphData.width} ${graphData.height}`}
          preserveAspectRatio="xMinYMin meet"
        >
          <defs>
            <marker
              id="proof-graph-arrowhead"
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
            >
              <path d="M 0 0 L 8 4 L 0 8 z" className="proof-graph-arrowhead" />
            </marker>
          </defs>

          {graphData.filteredMotoEdges.map((edge) => {
            const dependencyPosition = graphData.positions.get(edge.to);
            const proofPosition = graphData.positions.get(edge.from);
            if (!dependencyPosition || !proofPosition) {
              return null;
            }

            const startX = dependencyPosition.x + NODE_WIDTH;
            const startY = dependencyPosition.y + NODE_HEIGHT / 2;
            const endX = proofPosition.x;
            const endY = proofPosition.y + NODE_HEIGHT / 2;
            const controlOffset = Math.max((endX - startX) / 2, 24);
            const path = [
              `M ${startX} ${startY}`,
              `C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`,
            ].join(' ');

            return (
              <path
                key={`${edge.to}->${edge.from}`}
                d={path}
                className="proof-graph-edge"
                markerEnd="url(#proof-graph-arrowhead)"
              />
            );
          })}

          {graphData.filteredNodes.map((node) => {
            const position = graphData.positions.get(node.proof_id);
            if (!position) {
              return null;
            }

            const titleLines = wrapSvgText(
              node.theorem_name || node.theorem_statement || node.proof_id,
              30,
              3
            );
            const sourceLine = truncate(
              `${node.source_type} ${node.source_id}`.trim(),
              30
            );
            const mathlibCount = (graphData.mathlibByProofId.get(node.proof_id) || []).length;
            const isSelected = node.proof_id === expandedProofId;

            const handleActivate = () => {
              if (typeof onSelectProof === 'function') {
                onSelectProof(node.proof_id);
              }
            };

            const handleKeyDown = (event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                handleActivate();
              }
            };

            return (
              <g
                key={node.proof_id}
                className={`proof-graph-node-group ${node.is_novel ? 'novel' : 'known'} ${isSelected ? 'selected' : ''}`}
                transform={`translate(${position.x}, ${position.y})`}
                onClick={handleActivate}
                onKeyDown={handleKeyDown}
                onMouseEnter={() => setHoveredProofId(node.proof_id)}
                onMouseLeave={() => setHoveredProofId((current) => (current === node.proof_id ? null : current))}
                role="button"
                tabIndex={0}
              >
                <rect
                  className="proof-graph-node-frame"
                  width={NODE_WIDTH}
                  height={NODE_HEIGHT}
                  rx="18"
                  ry="18"
                />
                <text className="proof-graph-node-proof-id" x="16" y="24">
                  {node.proof_id}
                </text>
                {titleLines.map((line, index) => (
                  <text
                    key={`${node.proof_id}-line-${index}`}
                    className="proof-graph-node-title"
                    x="16"
                    y={48 + index * 18}
                  >
                    {line}
                  </text>
                ))}
                <text className="proof-graph-node-source" x="16" y={NODE_HEIGHT - 18}>
                  {sourceLine || 'Source unavailable'}
                </text>
                <text className="proof-graph-node-mathlib" x={NODE_WIDTH - 16} y="24" textAnchor="end">
                  {mathlibCount} mathlib
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export default ProofGraph;
