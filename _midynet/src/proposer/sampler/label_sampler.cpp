#include "FastMIDyNet/proposer/sampler/label_sampler.h"

namespace FastMIDyNet{

LabelPair LabelPairSampler::sample() const {
    double edgeWeights = m_edgeSampler.getTotalWeight();
    double vertexWeights = m_vertexSampler.getTotalWeight() * m_vertexSampler.getTotalWeight();
    double p = m_shift * vertexWeights / ( m_shift * vertexWeights + edgeWeights);

    LabelPair rs;
    if (m_bernoulliDistribution(rng) < p)
        rs = getLabelOfIdx({m_vertexSampler.sample(), m_vertexSampler.sample()});
    else
        rs = getLabelOfIdx(m_edgeSampler.sample());
    return rs;

}

}
