import argparse
from multiprocessing import cpu_count
from preprocess_utils.convert_csqa import convert_to_entailment
# from preprocess_utils.convert_piqa import convert_to_piqa_statement
from preprocess_utils.conceptnet import extract_english, construct_graph
from preprocess_utils.umls import construct_graph_umls
from preprocess_utils.grounding import create_matcher_patterns, ground
from preprocess_utils.grounding_umls import ground_umls
from preprocess_utils.graph import generate_adj_data_from_grounded_concepts
from preprocess_utils.graph_with_glove import generate_adj_data_from_grounded_concepts__use_glove
from preprocess_utils.graph_with_LM import generate_adj_data_from_grounded_concepts__use_LM
from preprocess_utils.graph_umls_with_glove import generate_adj_data_from_grounded_concepts_umls__use_glove


input_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'piqa': {
        'train': './data/piqa/raw/raw.train.jsonl',
        'dev': './data/piqa/raw/raw.dev.jsonl',
        'test': './data/piqa/raw/raw.test.jsonl',
    },
}


output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'umls': {
        'csv': './data/umls/umls.csv',
        'vocab': './data/umls/concepts.txt',
        'rel': './data/umls/relations.txt',
        'graph': './data/umls/umls.graph',
    },


    'piqa': {
        'statement': {
            'train': './data/piqa/statement/train.statement.jsonl',
            'dev': './data/piqa/statement/dev.statement.jsonl',
            'test': './data/piqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/piqa/grounded/train.grounded.jsonl',
            'dev': './data/piqa/grounded/dev.grounded.jsonl',
            'test': './data/piqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/piqa/graph/train.graph.adj.pk',
            'adj-dev': './data/piqa/graph/dev.graph.adj.pk',
            'adj-test': './data/piqa/graph/test.graph.adj.pk',
        },
    },
}

# for dname in ['medqa']:
#     output_paths[dname] = {
#         'statement': {
#             'train': f'./data/{dname}/statement/train.statement.jsonl',
#             'dev':   f'./data/{dname}/statement/dev.statement.jsonl',
#             'test':  f'./data/{dname}/statement/test.statement.jsonl',
#         },
#         'grounded': {
#             'train': f'./data/{dname}/grounded/train.grounded.jsonl',
#             'dev':   f'./data/{dname}/grounded/dev.grounded.jsonl',
#             'test':  f'./data/{dname}/grounded/test.grounded.jsonl',
#         },
#         'graph': {
#             'adj-train': f'./data/{dname}/graph/train.graph.adj.pk',
#             'adj-dev':   f'./data/{dname}/graph/dev.graph.adj.pk',
#             'adj-test':  f'./data/{dname}/graph/test.graph.adj.pk',
#         },
#     }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['piqa'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        
        'piqa': [
            {'func': convert_to_entailment, 'args': (input_paths['piqa']['train'], output_paths['piqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['piqa']['dev'], output_paths['piqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['piqa']['test'], output_paths['piqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['piqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['piqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['piqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['piqa']['grounded']['test'], args.nprocs)},
            {'func': ground, 'args': (output_paths['piqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['piqa']['grounded']['train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['piqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['piqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['piqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['piqa']['graph']['adj-test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['piqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['piqa']['graph']['adj-train'], args.nprocs)},
        ],
        
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
