import midynet
import numpy as np


def build_mcmc(N: int = 10, E: int = 25, Bmax: int = 10):
    config = midynet.config.RandomGraphConfig.hyperuniform_sbm(N, E, Bmax)
    random_graph = midynet.config.RandomGraphFactory.build(config)
    mcmc = midynet.config.RandomGraphMCMCFactory.build(config)
    callback = midynet.mcmc.callbacks.CollectPartitionOnSweep()
    mcmc.add_callback(callback)
    mcmc.set_random_graph(random_graph.get_wrap())
    mcmc.set_up()
    return midynet.config.Wrapper(
        mcmc,
        random_graph=random_graph,
        callback=callback,
        # **random_graph.get_others(),
        # **mcmc.get_others(),
    )


def main():
    wrap = build_mcmc(10, 20)
    mcmc = wrap.get_wrap()
    random_graph = wrap.get_other("random_graph")
    proposer = wrap.get_wrap().get_other("block_proposer")

    random_graph.sample()
    # print(random_graph.get_other("blocks").get_size())
    print(random_graph.get_blocks())
    # graph.sample()
    # random_graph.sample_blocks()
    # print(random_graph.get_blocks())

    for i in range(10000):
        move = proposer.propose_move()
        print(
            f"Move: [vertex {move.vertex_id}: {move.prev_block_id}({random_graph.get_block_of_idx(move.vertex_id)}) -> {move.next_block_id} (Blocks added: {move.added_blocks})]",
            end=", ",
        )
        dS = random_graph.get_log_joint_ratio(move)

        # dS = -random_graph.get_log_joint()
        # random_graph.apply_block_move(move)
        # dS += random_graph.get_log_joint()
        # random_graph.apply_block_move(reverse_move)

        dS += proposer.get_log_proposal_prob_ratio(move)
        print(f"dS: {dS:0.3f}")
        if dS > 1e4 or np.isnan(dS):
            print("Blocks: ", np.array(random_graph.get_blocks()))
            print("Block count : ", np.array(random_graph.get_block_count()))
            print("Vertex counts: ", np.array(random_graph.get_vertex_counts()))
            print("Edge Matrix:", np.array(random_graph.get_edge_matrix()))
            print("Edge counts: ", np.array(random_graph.get_edge_counts()))
            print("Joint ratio: ", random_graph.get_log_joint_ratio(move))
            print("Prop. ratio: ", proposer.get_log_proposal_prob_ratio(move))
            break
        accepted = False
        if np.exp(dS) > np.random.rand() and not np.isnan(dS):
            # mcmc.apply_block_move(move)
            # random_graph.apply_block_move(move)
            accepted = True
        print(f"Accepted: {accepted}")
        random_graph.check_self_consistency()
    # print(random_graph.get_blocks())
    # print(mcmc.do_MH_sweep(100))
    # print(random_graph.get_blocks())
    # # print(mcmc.do_MH_sweep(100))
    # # print(mcmc.do_MH_sweep(100))
    # # print(mcmc.do_MH_sweep(100))
    # print(graph.get_blocks())


if __name__ == "__main__":
    main()
