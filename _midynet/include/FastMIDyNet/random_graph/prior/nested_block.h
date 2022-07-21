#ifndef FAST_MIDYNET_NESTED_BLOCK_H
#define FAST_MIDYNET_NESTED_BLOCK_H

#include "block.h"

namespace FastMIDyNet{

class NestedBlockPrior: public BlockPrior{
protected:
    std::vector<std::vector<BlockIndex>> m_nestedState;
    NestedBlockCountPrior* m_nestedBlockCountPriorPtr = nullptr;
    std::vector<CounterMap<BlockIndex>> m_nestedVertexCounts;

    void _applyLabelMove(const BlockMove& move) override {
        m_nestedBlockCountPriorPtr->setNestedStateAtLevel(m_nestedBlockCountPriorPtr->getNestedStateAtLevel(move.level) + move.addedLabels, move.level);
        m_nestedVertexCounts[move.level].decrement(move.prevLabel);
        m_nestedVertexCounts[move.level].increment(move.nextLabel);

        if (move.level == 0){
            m_vertexCounts.decrement(move.prevLabel);
            m_vertexCounts.increment(move.nextLabel);
            m_state[move.vertexIndex] = move.nextLabel;
        }
    }


    const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override {
        std::vector<size_t> B = m_nestedBlockCountPriorPtr->getNestedState();
        double logLikelihoodBefore = m_nestedBlockCountPriorPtr->getLogLikelihoodFromNestedState(B);
        B[move.level] += move.addedLabels;
        double logLikelihoodAfter = m_nestedBlockCountPriorPtr->getLogLikelihoodFromNestedState(B);
        return logLikelihoodAfter - logLikelihoodBefore;
    }


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
    const BlockSequence& getNestedStateAtLevel(Level level) const { return m_nestedState[level]; }

    void setNestedState(const std::vector<std::vector<BlockIndex>>& nestedBlocks) {
        m_nestedVertexCounts = computeNestedVertexCounts(nestedBlocks);
        m_nestedBlockCountPriorPtr->setNestedStateFromNestedPartition(nestedBlocks);
        m_nestedState = nestedBlocks;
        setState(nestedBlocks[0]);
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
    const size_t getNestedBlockCountAtLevel(Level level) const {
        return (level==-1) ? getSize() : m_nestedBlockCountPriorPtr->getNestedStateAtLevel(level);
    }
    const size_t getNestedMaxBlockCountAtLevel(Level level) const {
        return (level==-1) ? getSize() : getMaxBlockCountFromPartition(getNestedStateAtLevel(level));
    }
    const std::vector<size_t> getNestedMaxBlockCount() const {
        std::vector<size_t> B;
        for (const auto& b: m_nestedState)
            B.push_back(getMaxBlockCountFromPartition(b));
        return B;
    }
    const size_t getNestedEffectiveBlockCountAtLevel(Level level) const {
        return (level==-1) ? getSize() : getEffectiveBlockCountFromPartition(getNestedStateAtLevel(level));
    }
    const std::vector<size_t> getNestedEffectiveBlockCount() const {
        std::vector<size_t> B;
        for (const auto& b: m_nestedState)
            B.push_back(getEffectiveBlockCountFromPartition(b));
        return B;
    }
    const std::vector<CounterMap<BlockIndex>>& getNestedVertexCounts() const { return m_nestedVertexCounts; };
    const CounterMap<BlockIndex>& getNestedVertexCountsAtLevel(Level l) const { return m_nestedVertexCounts[l]; };
    const BlockIndex getBlockOfIdx(BaseGraph::VertexIndex idx, Level level) const {
        if (level == -1)
            return (BlockIndex) idx;
        BlockIndex currentBlock = m_nestedState[0][idx];
        for (Level l=1; l<level; ++l)
            currentBlock = m_nestedState[l][currentBlock];
        return currentBlock;
    }
    static std::vector<CounterMap<BlockIndex>> computeNestedVertexCounts(const std::vector<std::vector<BlockIndex>>&);

    /* sampling methods */
    void sampleState() override{
        std::vector<BlockSequence> nestedBlocks;
        for (size_t l=0; l<getDepth(); ++l)
            nestedBlocks.push_back(sampleStateAtLevel(l));
        m_nestedState = nestedBlocks;
        m_nestedVertexCounts = computeNestedVertexCounts(m_nestedState);
        m_state = nestedBlocks[0];
        m_vertexCounts = m_nestedVertexCounts[0];
    }
    virtual const BlockSequence sampleStateAtLevel(Level level) const = 0;

    /* MCMC methods */
    const double getLogLikelihood() const override;
    virtual const double getLogLikelihoodAtLevel(Level level) const = 0;

    bool creatingNewBlock(const BlockMove& move) const {
        return creatingNewLevel(move) or m_nestedVertexCounts[move.level].get(move.nextLabel) == 0;
    }

    bool destroyingBlock(const BlockMove& move) const {
        return move.prevLabel != move.nextLabel and
               not creatingNewLevel(move) and
               m_nestedVertexCounts[move.level].get(move.prevLabel) == 1 ;
    }
    bool creatingNewLevel(const BlockMove& move) const {
        return move.level == m_nestedVertexCounts.size() - 1 and move.addedLabels == 1;
    }
    const int getAddedBlocks(const BlockMove& move) const {
        return (int) creatingNewBlock(move) - (int) destroyingBlock(move);
    }

    /* Consistency methods */
    void computationFinished() const override {
        m_isProcessed=false;
        m_nestedBlockCountPriorPtr->computationFinished();

    }

    void checkLevel(std::string prefix, Level level) const {
        if (level < -1 or level >= getDepth())
            throw std::logic_error(prefix + ": level "
                 + std::to_string(level) + " out of range [-1, "
                 + std::to_string(getDepth()) + "].");
    }

    void checkSelfConsistency() const override {
        m_nestedBlockCountPriorPtr->checkConsistency();
        for (Level l=0; l<getDepth()-1; ++l){
            std::string prefix = "NestedBlockPrior (l=" + std::to_string(l) + ")";
            checkBlockSequenceConsistencyWithVertexCounts(prefix, m_nestedState[l], m_nestedVertexCounts[l]);
            if (m_nestedState[l].size() != getNestedBlockCountAtLevel(l - 1))
                throw ConsistencyError(prefix + ": nested state (size "
                + std::to_string(m_nestedState[l].size()) +
                ") is inconsistent with block count (" + std::to_string(getNestedBlockCountAtLevel(l - 1)) +  ").");
            if (m_nestedVertexCounts[l].size() > getNestedBlockCountAtLevel(l)){
                throw ConsistencyError(prefix + ": nested vertex counts (size "
                + std::to_string(m_nestedVertexCounts[l].size()) +
                ") are inconsistent with block count (" + std::to_string(getNestedBlockCountAtLevel(l)) +  ").");
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
    const BlockSequence sampleStateAtLevel(Level level) const override;
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
    const BlockSequence sampleStateAtLevel(Level level) const override;
};


}

#endif
