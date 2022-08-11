#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/random_graph/util.h"


namespace FastMIDyNet{

MultiGraph getLabelGraphFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq){
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
    MultiGraph edgeMat(numBlocks);
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            if (idx > neighbor.vertexIndex)
                continue;
            BlockIndex r = blockSeq[idx], s = blockSeq[neighbor.vertexIndex];
            edgeMat.addMultiedgeIdx(r, s, neighbor.label);
        }
    }
    return edgeMat;
};

void checkGraphConsistencyWithLabelGraph(
    std::string namePrefix,
    const MultiGraph& graph,
    const BlockSequence& blockSeq,
    const MultiGraph& expectedEdgeMat){
    MultiGraph actualEdgeMat = getLabelGraphFromGraph(graph, blockSeq);
    for (auto r : actualEdgeMat)
        for (auto s : actualEdgeMat.getNeighboursOfIdx(r))
            if (expectedEdgeMat.getEdgeMultiplicityIdx(r, s.vertexIndex) != s.label)
                throw ConsistencyError(
                    namePrefix,
                    "label graph", "edgeCount=" + std::to_string(s.label),
                    "graph", "edgeCount=" + std::to_string(expectedEdgeMat.getEdgeMultiplicityIdx(r, s.vertexIndex)),
                    "(r=" + std::to_string(r) + ", s=" + std::to_string(s.vertexIndex) + ")"
                );
};

void checkGraphConsistencyWithDegreeSequence(std::string className, std::string expName, const MultiGraph& graph, std::string actName, const DegreeSequence& expectedDegreeSeq){
    DegreeSequence actualDegreeSeq = graph.getDegrees();

    for (auto idx : graph){
        if (expectedDegreeSeq[idx] != actualDegreeSeq[idx])
            throw ConsistencyError(
                className,
                expName, "k=" + std::to_string(expectedDegreeSeq[idx]),
                actName, "k=" + std::to_string(actualDegreeSeq[idx]),
                "vertex=" + std::to_string(idx)
            );
    }
}

EdgeCountPrior* makeEdgeCountPrior(double edgeCount, bool canonical){
    if (canonical)
        return new EdgeCountExponentialPrior(edgeCount) ;
    else
        return new EdgeCountDeltaPrior((size_t) edgeCount) ;
}

BlockPrior* makeBlockPrior(size_t size, BlockCountPrior& blockCountPrior, bool hyperPrior){
    if (hyperPrior)
        return new BlockUniformHyperPrior(size, blockCountPrior) ;
    else
        return new BlockUniformPrior(size, blockCountPrior) ;
}

DegreePrior* makeDegreePrior(size_t size, EdgeCountPrior& prior, bool hyperPrior){
    if (hyperPrior)
        return new DegreeUniformHyperPrior(size, prior) ;
    else
        return new DegreeUniformPrior(size, prior) ;
}

VertexLabeledDegreePrior* makeVertexLabeledDegreePrior(LabelGraphPrior& prior, bool hyperPrior){
    if (hyperPrior)
        return new VertexLabeledDegreeUniformHyperPrior(prior) ;
    else
        return new VertexLabeledDegreeUniformPrior(prior) ;
}

StochasticBlockModelLikelihood* makeSBMLikelihood(bool stubLabeled){
    if (stubLabeled)
        return new StubLabeledStochasticBlockModelLikelihood() ;
    else
        return new UniformStochasticBlockModelLikelihood() ;
}

EdgeProposer* makeEdgeProposer(
    std::string proposerType,
    bool canonical,
    bool degreeConstrained,
    // bool labelConstrained=false,
    bool withSelfLoops,
    bool withParallelEdges){
    if ( canonical ){
        if ( proposerType == "uniform" )
            return new SingleEdgeUniformProposer(withSelfLoops, withParallelEdges);
        else if ( proposerType == "degree" )
                return new SingleEdgeUniformProposer(withSelfLoops, withParallelEdges);
        else
            throw std::runtime_error("makeEdgeProposer: invalid proposer type `" + proposerType + "`.");
    } else if ( not degreeConstrained ){
        if ( proposerType == "uniform" )
            return new HingeFlipUniformProposer(withSelfLoops, withParallelEdges);
        else if ( proposerType == "degree" )
                return new HingeFlipUniformProposer(withSelfLoops, withParallelEdges);
        else
            throw std::runtime_error("makeEdgeProposer: invalid proposer type `" + proposerType + "`.");
    } else
        return new DoubleEdgeSwapProposer(withSelfLoops, withParallelEdges);
}

LabelProposer<BlockIndex>* makeBlockProposer(
    std::string proposerType, bool restricted, double sampleLabelCountProb, double labelCreationProb, double shift
){
    if (proposerType == "uniform"){
        if (restricted)
            return new RestrictedUniformBlockProposer(sampleLabelCountProb);
        else
            return new GibbsUniformBlockProposer(sampleLabelCountProb, labelCreationProb);
    } else if (proposerType == "mixed"){
        if (restricted)
            return new RestrictedMixedBlockProposer(sampleLabelCountProb, shift);
        else
            return new GibbsMixedBlockProposer(sampleLabelCountProb, labelCreationProb, shift);
    }
    else
        throw std::runtime_error("makeBlockProposer: invalid proposer type `" + proposerType + "`.");
}

NestedLabelProposer<BlockIndex>* makeNestedBlockProposer(
    std::string proposerType, bool restricted, double sampleLabelCountProb, double labelCreationProb, double shift
){
    if (proposerType == "uniform"){
        if (restricted)
            return new RestrictedUniformNestedBlockProposer(sampleLabelCountProb);
        else
            return new GibbsUniformNestedBlockProposer(sampleLabelCountProb, labelCreationProb);
    } else if (proposerType == "mixed"){
        if (restricted)
            return new RestrictedMixedNestedBlockProposer(sampleLabelCountProb, shift);
        else
            return new GibbsMixedNestedBlockProposer(sampleLabelCountProb, labelCreationProb, shift);
    }
    else
        throw std::runtime_error("makeNestedBlockProposer: invalid proposer type `" + proposerType + "`.");
}


}
