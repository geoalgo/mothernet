import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mothernet.prediction.mothernet import EnsembleMeta, MotherNetClassifier, ShiftClassifier
from mothernet.prediction.tabpfn import TabPFNClassifier

MOTHERNET_PATH = "models_diff/prior_diff_real_checkpointcontinue_hidden_128_embed_dim_1024_decoder_nhid_2048_nlayer12_lr0003_n_0_epoch_on_exit.cpkt"
MOTHERNET_L2_PATH = "models_diff/mothernet_128_decoder_2048_emsize_512_nlayers_12_steps_8192_bs_8ada_lr_3e-05_1_gpu_07_31_2023_23_18_33_epoch_780.cpkt"
MOTHERNET_LOW_RANK_PATH = "models_diff/mn_n1024_L2_W128_P512_1_gpu_08_03_2023_03_48_19_epoch_on_exit.cpkt"
MOTHERNET_NEW_CODE = "models_diff/mn_d2048_H4096_1_gpu_08_04_2023_16_28_11_epoch_950_kept_for_testing.cpkt"
TABPFN_MODEL_PATH = "models_diff/download_epoch_100.cpkt"


@pytest.mark.parametrize("ensemble", [ShiftClassifier, EnsembleMeta, None])
@pytest.mark.parametrize("class_offset", [0, 4])
def test_basic_iris(ensemble, class_offset):
    pytest.skip("haven't checked in model checkpoints yet")

    if class_offset == 4 and ensemble in [None, ShiftClassifier]:
        raise pytest.skip("Skip this test because the ensemble is None and class_offset is 4")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = MotherNetClassifier(path=MOTHERNET_PATH, device='cpu')
    if ensemble is not None:
        mothernet = ensemble(mothernet)
    mothernet.fit(X_train, y_train + class_offset)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test + class_offset) > 0.9


def test_predict_new_training_code_iris():
    pytest.skip("haven't checked in model checkpoints yet")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = MotherNetClassifier(path=MOTHERNET_NEW_CODE, device='cpu')
    mothernet.fit(X_train, y_train)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test) > 0.9


def test_two_layers_iris():
    pytest.skip("haven't checked in model checkpoints yet")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = MotherNetClassifier(path=MOTHERNET_L2_PATH, device='cpu')
    mothernet.fit(X_train, y_train)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test) > 0.9


def test_two_layers_iris_tabpfn_logic():
    pytest.skip("haven't checked in model checkpoints yet")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = TabPFNClassifier(model_string="mothernet_128_decoder_2048_emsize_512_nlayers_12_steps_8192_bs_8ada_lr_3e-05_1_gpu_07_31_2023_23_18_33",
                                 epoch=780, device='cpu', N_ensemble_configurations=1)
    mothernet.fit(X_train, y_train)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test) > 0.9


def test_low_rank_iris():
    pytest.skip("haven't checked in model checkpoints yet")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = MotherNetClassifier(path=MOTHERNET_LOW_RANK_PATH, device='cpu')
    mothernet.fit(X_train, y_train)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test) > 0.9


def test_low_rank_iris_tabpfn_logic():
    pytest.skip("haven't checked in model checkpoints yet")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = TabPFNClassifier(model_string="mn_n1024_L2_W128_P512_1_gpu_08_03_2023_03_48_19", epoch="on_exit", device='cpu')
    mothernet.fit(X_train, y_train)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test) > 0.9


def test_tabpfn_load_error():
    pytest.skip("haven't checked in model checkpoints yet")
    iris = load_iris()
    # test that we get a good error if we try to load tabpfn weights in MotherNetClassifier
    mothernet = MotherNetClassifier(path=TABPFN_MODEL_PATH, device='cpu')
    with pytest.raises(ValueError, match="Cannot load tabpfn weights into MotherNetClassifier"):
        mothernet.fit(iris.data, iris.target)