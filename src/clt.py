import argparse 
import sys
import json
import os
import pandas as pd
from src.base import compute_tsne_datasets, get_tsne_plots_perplexities, get_tsne_plots_groups, get_tsne_plot, is_previously_computed, get_filters_with_key, apply_filters

def process_arguments():
    parser = argparse.ArgumentParser(
        prog="t-SNE.py",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description="A tool that performs t-SNE and creates plots for analysis"
    )

    parser.add_argument(
        '--key',
        type=str,
        help='t-SNE keyname',
        required= True
    )

    parser.add_argument(
        '--action',
        choices = ['compute', 'plot', 'interact'],
        required= True
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help="Specifies the input file in CSV format"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help= "Specified the directory where the tool's output will be saved"
    )
    parser.add_argument(
        '--columns_exclude',
        type=str,
        nargs = '+',
        help= "Array of columns which should be excluded when performing t-SNE",
    )

    parser.add_argument(
        '--perplexities',
        type= int,
        help= "Array of perplexities that should be computed. If t-SNE if already computed for a value, will be ignored"
    )

    parser.add_argument(
        "--learning_rate",
        type=str,
        help = "Acceptable values for learning rate are auto, or a flaot value. The range of the float value must be between 10 to 1000"
    )

    parser.add_argument(
        '--early_exaggeration',
        type=float,
        help = "Acceptable values for learning rate are float values from 1-20"
    )

    parser.add_argument(
        "--n_iter",
        type = int,
        help = "N iter"
    )

    parser.add_argument(
        "--n_components_pca",
        type = int,
        help = "Number of PCA compoinents to select before applying t-SNE"
    )

    parser.add_argument(
        "--group_columns",
        type = str,
        nargs = '+',
        help = "Which column to group by and select the data",
        default=None
    )

    parser.add_argument(
        "--hover_columns",
        type = str,
        help = "Which column to group by and select the data",
        default= None
    )

    parser.add_argument(
        '--parallel',
        action = 'store_true',
        help = 'Generate a parralelizable script instead of running on same machine'
    )

    parser.add_argument(
        "--selected",
        nargs = '+',
        type = str,
        help = 'Cluster name which you want to plot.'
    )

    args = parser.parse_args(sys.argv[1:])
    return args

def set_default_args(args, conf):
    for arg in vars(args):
        if getattr(args, arg) is None and arg in conf:
            setattr(args, arg, conf[arg])

    return args

def main():
    # TODO: Work on logging all actions
    # TODO: Add checks for every edge case

    args = process_arguments()

    conf_dir = os.getcwd() + f'/keys/{args.key}/conf.json'
    output_dir = os.getcwd() + f'/keys/{args.key}/tSNE_data/'

    f = open(conf_dir)

    conf = json.load(f)
    

    args = set_default_args(args, conf)
    if args.action == 'compute':
        input_df = pd.read_csv(conf['input_file'])

        filters = get_filters_with_key(args.key)
        if filters is not None and len(filters):
            input_df = apply_filters(input_df, filters)

        if args.parallel:
            f = open("00master_tsne.sh", "w")
            pwd = os.getcwd()
            for perp in args.perplexities:
                if is_previously_computed(output_dir, perp):
                    print(f"t_SNE has already been computed for perplexity = {perp}, skipping computation.")

                else:
                    f.writelines([f'python {pwd}/clt.py --action compute --input_file {conf["input_file"]} --output_dir {output_dir} --perplexities {perp} --columns_exclude {" ".join(conf["columns_exclude"])} --learning_rate {conf["learning_rate"]} --early_exaggeration {conf["early_exaggeration"]}  --n_iter {conf["n_iter"]} --n_components_pca {conf["n_components_pca"]}\n'])
            print("Generated 00master script. Run it to compute TSNE values for above mentioned perplexities")

        else:
            for perp in args.perplexities:
                if is_previously_computed(output_dir, perp):
                    print(f"t=SNE has already been computed for perplexity = {perp}, skipping computation")
                else:
                    compute_tsne_datasets(input_df, output_dir, args.columns_exclude, args.n_components_pca, args.perplexities, args.n_iter, args.learning_rate)

    elif args.action == 'plot':

        perp = args.perplexities
        for perp in args.perplexities:
            if not is_previously_computed(output_dir, perp):
                print(f"t-SNE values for perplexity = {perp} has not been computed. Compute the values using --action compte and then plot. ")
                return
        
        if args.perplexities and len(args.perplexities) > 1 and args.group_columns and len(args.group_columns) > 1:
            print("Can't have multiple perplexities AND group columns. Make one of them singular to plot")
        elif args.perplexities and len(args.perplexities) > 1:
            get_tsne_plots_perplexities(output_dir, perplexities = args.perplexities, group_columns = args.group_columns, selected_groups = args.selected, s = 0.05)
        elif args.group_columns and len(args.group_columns) > 1:
            get_tsne_plots_groups(output_dir, perplexity = args.perplexities[0], group_columns = args.group_columns, s = 0.05)
        elif args.perplexities and len(args.perplexities) == 1:
            get_tsne_plot(output_dir, args.perplexities, args.group_columns, args.selected)

    if __name__ == '__main__':
        main()






