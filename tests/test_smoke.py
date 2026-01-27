def test_imports():
    import torch
    import timm
    import mlflow
    import dvc


def test_model_init():
    from radar_mlops import MultimodalModel
    model = MultimodalModel(num_classes=9)
    assert model is not None
