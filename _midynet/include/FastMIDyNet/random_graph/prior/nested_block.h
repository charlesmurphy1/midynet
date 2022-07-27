#ifndef FAST_MIDYNET_NESTED_BLOCK_H
#define FAST_MIDYNET_NESTED_BLOCK_H

#include "block.h"

namespace FastMIDyNet{

class NestedBlockPrior: public BlockPrior{
protected:
    NestedBlockCountPrior* m_nestedBlockCountPriorPtr = nullptr;
    std::vector<std::vector<BlockIndex>> m_nestedState;
    std::vector<CounterMap<BlockIndex>> m_nestedVertexCounts;
    std::vector<CounterMap<BlockIndex>> m_nestedAbsVertexCounts;

    void _applyLabelMove(const BlockMove& move) override ;
    const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override ;

    // void remapBlockIndex(const std::map<size_t, size_t> indexMap){
    //     auto newBlocks = m_state;
    //     for (size_t v=0; v<m_size; ++v){
    //         newBlocks[v] = indexMap.at(m_state[v]);
    //     }
    //     setState(newBlocks);
    // }

public:
    /* Constructors */
    NestedBlockPrior() {}
    NestedBlockPrior(size_t size, NestedBlockCountPrior& blockCountPrior):
        BlockPrior(size, blockCountPrior) { setNestedBlockCountPrior(blockCountPrior); }
    NestedBlockPrior(const NestedBlockPrior& other) {
        setSize(other.m_size);
        setNestedState(other.m_nestedState);
        this->setNestedBlockCountPrior(*other.m_nestedBlockCountPriorPtr);
    }
    virtual ~NestedBlockPrior(){}
    const NestedBlockPrior& operator=(const NestedBlockPrior& other){
        setNestedState(other.m_nestedState);
        this->setNestedBlockCountPrior(*other.m_nestedBlockCountPriorPtr);
        return *this;
    }

    const std::vector<BlockSequence>& getNestedState() const { return m_nestedState; }
    const BlockSequence& getNestedState(Level level) const { return m_nestedState[level]; }

    void setNestedState(const std::vector<std::vector<BlockIndex>>& nestedBlocks) {
        m_nestedState = nestedBlocks;
        m_nestedVertexCounts = computeNestedVertexCounts(m_nestedState);
        m_nestedAbsVertexCounts = computeNestedAbsoluteVertexCounts(m_nestedState);
        m_nestedBlockCountPriorPtr->setNestedStateFromNestedPartition(m_nestedState);
        setState(m_nestedState[0]);
    }

    /* Accessors & mutators of attributes */
    const size_t getSize() const { return m_size; }
    virtual void setSize(size_t size) override { m_size = size; }

    const size_t getDepth() const { return m_nestedBlockCountPriorPtr->getDepth(); }

    /* Accessors & mutators of accessory states */
    const NestedBlockCountPrior& getNestedBlockCountPrior() const { return *m_nestedBlockCountPriorPtr; }
    NestedBlockCountPrior& getNestedBlockCountPriorRef() const { return *m_nestedBlockCountPriorPtr; }
    void setNestedBlockCountPrior(NestedBlockCountPrior& prior) {
        setBlockCountPrior(prior);
        m_nestedBlockCountPriorPtr = &prior;
        m_nestedBlockCountPriorPtr->isRoot(false);
    }

    const std::vector<size_t>& getNestedBlockCount() const { return m_nestedBlockCountPriorPtr->getNestedState(); }
    const size_t getNestedBlockCount(Level level) const {
        return (level==-1) ? getSize() : m_nestedBlockCountPriorPtr->getNestedState(level);
    }
    const size_t getNestedMaxBlockCount(Level level) const {
        return (level==-1) ? getSize() : getMaxBlockCountFromPartition(getNestedState(level));
    }
    const std::vector<size_t> getNestedMaxBlockCount() const {
        std::vector<size_t> B;
        for (const auto& b: m_nestedState)
            B.push_back(getMaxBlockCountFromPartition(b));
        return B;
    }
    const size_t getNestedEffectiveBlockCount(Level level) const {
        return (level==-1) ? getSize() : getNestedAbsVertexCounts(level).size();
    }
    const std::vector<size_t> getNestedEffectiveBlockCount() const {
        std::vector<size_t> B;
        for (Level level=0; level<getDepth(); ++level)
            B.push_back(getNestedEffectiveBlockCount(level));
        return B;
    }
    const std::vector<CounterMap<BlockIndex>>& getNestedVertexCounts() const { return m_nestedVertexCounts; };
    const CounterMap<BlockIndex>& getNestedVertexCounts(Level l) const { return m_nestedVertexCounts[l]; };
    const std::vector<CounterMap<BlockIndex>>& getNestedAbsVertexCounts() const { return m_nestedAbsVertexCounts; };
    const CounterMap<BlockIndex>& getNestedAbsVertexCounts(Level l) const { return m_nestedAbsVertexCounts[l]; };
    const BlockIndex getBlockOfIdx(BaseGraph::VertexIndex idx, Level level) const {
        if (level == -1)
            return (BlockIndex) idx;
        if (level == getDepth())
            return 0;
        BlockIndex currentBlock = idx;
        for (Level l=0; l<=level; ++l)
            currentBlock = m_nestedState[l][currentBlock];
        return currentBlock;
    }
    const BlockIndex getNestedBlockOfIdx(BlockIndex idx, Level level) const { return m_nestedState[level][idx]; }
    static std::vector<CounterMap<BlockIndex>> computeNestedVertexCounts(const std::vector<std::vector<BlockIndex>>&);
    static std::vector<CounterMap<BlockIndex>> computeNestedAbsoluteVertexCounts(const std::vector<std::vector<BlockIndex>>&);
    static std::vector<BlockSequence> reduceHierarchy(const std::vector<BlockSequence>& nestedState) ;
    void reduceHierarchy() { setNestedState(reduceHierarchy(m_nestedState)); }

    /* sampling methods */
    void sampleState() override{
        std::vector<BlockSequence> nestedBlocks;
        for (size_t l=0; l<getDepth(); ++l)
            nestedBlocks.push_back(sampleState(l));
        m_nestedState = nestedBlocks;
        m_nestedVertexCounts = computeNestedVertexCounts(m_nestedState);
        m_nestedAbsVertexCounts = computeNestedAbsoluteVertexCounts(m_nestedState);
        m_state = nestedBlocks[0];
        m_vertexCounts = m_nestedVertexCounts[0];
    }
    virtual const BlockSequence sampleState(Level level) const = 0;

    /* MCMC methods */
    const double getLogLikelihood() const override;
    virtual const double getLogLikelihoodAtLevel(Level level) const = 0;

    bool creatingNewBlock(const BlockMove& move) const {
        return creatingNewLevel(move) or m_nestedVertexCounts[move.level].get(move.nextLabel) == 0;
    }

    bool destroyingBlock(const BlockMove& move) const {
        return move.prevLabel != move.nextLabel and not creatingNewLevel(move) and m_nestedVertexCounts[move.level].get(move.prevLabel) == 1;
    }
    bool creatingNewLevel(const BlockMove& move) const {
        return move.level == m_nestedVertexCounts.size() - 1 and move.addedLabels == 1;
    }
    virtual void createNewBlock(const BlockMove& move) ;
    virtual void destroyBlock(const BlockMove& move) { };
    const int getAddedBlocks(const BlockMove& move) const {
        return (int) creatingNewBlock(move) - (int) destroyingBlock(move);
    }

    /* Consistency methods */
    void computationFinished() const override {
        m_isProcessed=false;
        m_nestedBlockCountPriorPtr->computationFinished();

    }

    void checkLevel(std::string prefix, Level level) const {
        if (level < -1 or level >= (int) getDepth())
            throw std::logic_error(prefix + ": level "
                 + std::to_string(level) + " out of range [-1, "
                 + std::to_string(getDepth()) + "].");
    }

    bool isValidBlockMove(const BlockMove& move) const ;

    void checkNestedStateConsistencyWithAbsVertexCounts() const {
        std::vector<CounterMap<size_t>> actualNestedAbsVertexCount = computeNestedAbsoluteVertexCounts(m_nestedState);
        for (Level l=0; l<getDepth(); ++l){
            std::string prefix = "NestedBlockPrior (l=" + std::to_string(l) + ")";
            size_t N = 0;
            for (const auto& nr : m_nestedAbsVertexCounts[l]){
                if (nr.second != actualNestedAbsVertexCount[l][nr.first])
                    throw ConsistencyError(prefix + ": nested state is inconsistent with absolute vertex counts at r="
                        + std::to_string(nr.first) + ": expected=" + std::to_string(nr.second) + " actual="
                        + std::to_string(actualNestedAbsVertexCount[l][nr.first]) + ".");
                N += nr.second;
            }
            if (N != getSize())
                throw ConsistencyError(prefix + ": nested state (size=" + std::to_string(getSize())
                    + ") is inconsistent with absolute vertex counts (size=" + std::to_string(N) + ").");
        }
    }

    void checkSelfConsistency() const override {
        m_nestedBlockCountPriorPtr->checkConsistency();
        checkNestedStateConsistencyWithAbsVertexCounts();
        for (Level l=0; l<getDepth()-1; ++l){
            std::string prefix = "NestedBlockPrior (l=" + std::to_string(l) + ")";
            checkBlockSequenceConsistencyWithVertexCounts(prefix, m_nestedState[l], m_nestedVertexCounts[l]);
            if (m_nestedState[l].size() < getNestedBlockCount(l - 1))
                throw ConsistencyError(prefix + ": nested state (size "
                + std::to_string(m_nestedState[l].size()) +
                ") is inconsistent with block count (" + std::to_string(getNestedBlockCount(l - 1)) +  ").");
            if (m_nestedVertexCounts[l].size() > getNestedBlockCount(l)){
                throw ConsistencyError(prefix + ": nested vertex counts (size "
                + std::to_string(m_nestedVertexCounts[l].size()) +
                ") are inconsistent with block count (" + std::to_string(getNestedBlockCount(l)) +  ").");
            }
        }
    }

    bool isSafe() const override {
        return (m_size != 0) and (m_nestedBlockCountPriorPtr != nullptr) and (m_nestedBlockCountPriorPtr->isSafe());
    }
    void checkSelfSafety() const override {
        if (m_size == 0)
            throw SafetyError("NestedBlockPrior: unsafe prior since `m_size` is zero.");
        if (m_nestedBlockCountPriorPtr == nullptr)
            throw SafetyError("NestedBlockPrior: unsafe prior since `m_nestedBlockCountPriorPtr` is empty.");
        m_nestedBlockCountPriorPtr->checkSafety();

    }

};

class NestedBlockUniformPrior: public NestedBlockPrior{
    NestedBlockCountUniformPrior m_nestedBlockCountPrior;
public:
    NestedBlockUniformPrior(size_t graphSize):
        NestedBlockPrior(),
        m_nestedBlockCountPrior(graphSize) {
            setSize(graphSize);
            setNestedBlockCountPrior(m_nestedBlockCountPrior);
        }
    virtual ~NestedBlockUniformPrior() { }

    void setSize(size_t size) override { m_size = size; m_nestedBlockCountPrior.setGraphSize(size); }
    const double getLogLikelihoodAtLevel(Level level) const override;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const ;
    const BlockSequence sampleState(Level level) const override;
};

class NestedBlockUniformHyperPrior: public NestedBlockPrior{
    NestedBlockCountUniformPrior m_nestedBlockCountPrior;
public:
    NestedBlockUniformHyperPrior(size_t graphSize):
        NestedBlockPrior(),
        m_nestedBlockCountPrior(graphSize) {
            setSize(graphSize);
            setNestedBlockCountPrior(m_nestedBlockCountPrior);
        }
    virtual ~NestedBlockUniformHyperPrior() { }

    void setSize(size_t size) override { m_size = size; m_nestedBlockCountPrior.setGraphSize(size); }
    const double getLogLikelihoodAtLevel(Level level) const override;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const ;
    const BlockSequence sampleState(Level level) const override;

    void destroyBlock(const BlockMove& move) override {
        BlockIndex r = getNestedBlockOfIdx(move.prevLabel, move.level + 1);
        m_nestedVertexCounts[move.level + 1].decrement(r);
        reduceHierarchy();
    }

};


}

#endif
