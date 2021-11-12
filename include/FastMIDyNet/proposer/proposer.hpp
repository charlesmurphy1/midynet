#ifndef FAST_MIDYNET_PROPOSER_HPP
#define FAST_MIDYNET_PROPOSER_HPP


namespace FastMIDyNet{
    template <typename T>
    class Proposer{
    public:
        Proposer() { }
        virtual const T& operator()() const = 0;
        virtual const double& getProposalProb(const T&) = 0;
        virtual const double& applyMove(const T&) = 0;

    protected:

    };
}

#endif
