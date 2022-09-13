import pdb
import click
from utils import load_dataset, get_split, save_model
from methods import get_model, evaluate_model


@click.command()
@click.option('--dataset', default='glasgow', help='dataset')
@click.option('--modelname',help='which model of {rf, lr} \
should be used, if None, all are used')
@click.option('--common_with')
@click.option('--label_class', default="label_all")
@click.option('--model_path')
@click.option('--evaluate_only', default=False, is_flag=True)
@click.option('--rf_max_depth', type=int)
def run_benchmark(dataset, modelname, common_with, label_class, model_path, evaluate_only, rf_max_depth):
    X_train, X_valid, X_test, y_train, y_valid, y_test, _ = get_split(dataset, label_class, common_with)

    # get model, load from disk if model_path is specified
    
    model = get_model(modelname, model_path, rf_max_depth)
    
    if not evaluate_only:
        print("fit model..")
        model.fit(X_train, y_train.astype("int"))
    modelparams = {'rf_max_depth':rf_max_depth}
    
    val_auc = evaluate_model(model, X_test, y_test, label_class, modelname, dataset, save=(not evaluate_only), common_with=common_with, modelparams=modelparams)
    test_auc = evaluate_model(model, X_valid, y_valid, label_class, modelname, dataset, save=(not evaluate_only), common_with=common_with, modelparams=modelparams)

    results = {'modelname' : modelname, 'label_class' : label_class, 'params' : modelparams, 'val_auc': val_auc, 'test_auc':test_auc}
    print(results)
    if not evaluate_only:
        save_model(model, results, name=modelname, dataset=dataset, label_class=label_class, common_with=common_with)

if __name__ == '__main__':
    run_benchmark()
