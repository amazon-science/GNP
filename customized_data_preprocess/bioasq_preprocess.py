import argparse
from multiprocessing import cpu_count
from preprocess_utils.convert_csqa import convert_to_entailment
# from preprocess_utils.convert_bioasq import convert_to_bioasq_statement
from preprocess_utils.conceptnet import extract_english, construct_graph
from preprocess_utils.umls import construct_graph_umls
from preprocess_utils.grounding import create_matcher_patterns, ground
from preprocess_utils.grounding_umls import ground_umls
from preprocess_utils.graph import generate_adj_data_from_grounded_concepts
from preprocess_utils.graph_with_glove import generate_adj_data_from_grounded_concepts__use_glove
from preprocess_utils.graph_with_LM import generate_adj_data_from_grounded_concepts__use_LM
from preprocess_utils.graph_umls_with_glove import generate_adj_data_from_grounded_concepts_umls__use_glove


input_paths = {
    # 'cpnet': {
    #     'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    # },
    'bioasq': {
        'train': './data/bioasq/raw/raw.train.jsonl',
        'dev': './data/bioasq/raw/raw.dev.jsonl',
        'test': './data/bioasq/raw/raw.test.jsonl',
    },
}


output_paths = {
    # 'cpnet': {
    #     'csv': './data/cpnet/conceptnet.en.csv',
    #     'vocab': './data/cpnet/concept.txt',
    #     'patterns': './data/cpnet/matcher_patterns.json',
    #     'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
    #     'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    # },
    'umls': {
        'csv': './data/umls/umls.csv',
        'vocab': './data/umls/concepts.txt',
        'rel': './data/umls/relations.txt',
        'graph': './data/umls/umls.graph',
    },


    'bioasq': {
        'statement': {
            'train': './data/bioasq/statement/train.statement.jsonl',
            'dev': './data/bioasq/statement/dev.statement.jsonl',
            'test': './data/bioasq/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/bioasq/grounded/train.grounded.jsonl',
            'dev': './data/bioasq/grounded/dev.grounded.jsonl',
            'test': './data/bioasq/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/bioasq/graph/train.graph.adj.pk',
            'adj-dev': './data/bioasq/graph/dev.graph.adj.pk',
            'adj-test': './data/bioasq/graph/test.graph.adj.pk',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['bioasq'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        
        'bioasq': [
            {'func': convert_to_entailment, 'args': (input_paths['bioasq']['train'], output_paths['bioasq']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['bioasq']['dev'], output_paths['bioasq']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['bioasq']['test'], output_paths['bioasq']['statement']['test'])},
            {'func': ground_umls, 'args': (output_paths['bioasq']['statement']['dev'], output_paths['umls']['vocab'], output_paths['bioasq']['grounded']['dev'], args.nprocs)},
            {'func': ground_umls, 'args': (output_paths['bioasq']['statement']['test'], output_paths['umls']['vocab'], output_paths['bioasq']['grounded']['test'], args.nprocs)},
            {'func': ground_umls, 'args': (output_paths['bioasq']['statement']['train'], output_paths['umls']['vocab'], output_paths['bioasq']['grounded']['train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (output_paths['bioasq']['grounded']['test'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['bioasq']['graph']['adj-test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (output_paths['bioasq']['grounded']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['bioasq']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (output_paths['bioasq']['grounded']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['bioasq']['graph']['adj-train'], args.nprocs)},
        ],
        
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
