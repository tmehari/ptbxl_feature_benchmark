import pdb
import click
from utils import load_dataset, get_split
from methods import get_model, evaluate_model


@click.command()
@click.option('--dataset', default='glasgow', help='dataset')
@click.option('--modelname',help='which model of {rf, lr} \
should be used, if None, all are used')
@click.option('--common_with', default='ge')
@click.option('--label_class', default="label_all")
@click.option('--model_path')
@click.option('--test_only', default=False, is_flag=True)
def run_benchmark(dataset, modelname, common_with, label_class, model_path, test_only):
    X_train, X_test, y_train, y_test = get_split(dataset, label_class, common_with)

    # get model, load from disk if model_path is specified
    model = get_model(modelname, model_path)
    
    if not test_only:
        print("fit model..")
        model.fit(X_train, y_train.astype("int"))
        
    evaluate_model(model, X_test, y_test, label_class, modelname, save=(not test_only))

if __name__ == '__main__':
    run_benchmark()
