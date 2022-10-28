# SegmentationEvaluator

A tool that will help to evaluate metrics from the inference of your model.

Make sure that you have the torch.jit.script model loaded with state_dict.
Moreover, make sure there exists the dataset that you are going to evaluate your model with it.

```code{
git clone https://github.com/mertalifirat/SegmentationEvaluator 

pip install -r requirements.txt

```
It is time to edit the main.py where described as TODO in main.py.

After that you are done, just execute the following line:
```code{
python main.py
```
Note that for now only precision,recall and IoU metrics can be calculated.
UPDATE
Creating mask option is also added.

Reference https://github.com/open-mmlab/mmsegmentation
