# BackdoorVault

BackdoorVault is a toolbox that collects representative backdoor attacks. It is based on PyTorch and still under development.

Some part of the implementation is drawn from the original backdoor papers. Thank authors for their open-sourced implemetations.

## Prerequisite

The code is implemented and tested on PyTorch. It runs on Python 3.6.9.

## Usage

The main functions are located in `main.py` file. For a test drive, please use the following command:

   ```bash
   python3 main.py --phase poison
   ```

This will generate a backdoored model by a polygon trigger with target label 0.

## Backdoor Attacks

The following attacks have been included in this toolbox.

[BadNets](https://arxiv.org/abs/1708.06733), [Blend](https://arxiv.org/abs/1712.05526), [DFST](https://arxiv.org/abs/2012.11212), [Dynamic](https://arxiv.org/abs/2003.03675), [Filter](https://www.cs.purdue.edu/homes/taog/docs/CCS19.pdf), [Input-aware](https://arxiv.org/abs/2010.08138), [Invisible](https://arxiv.org/abs/2012.03816), [Refool](https://arxiv.org/abs/2007.02343), [SIG](https://arxiv.org/abs/1902.11237), [WaNet](https://arxiv.org/abs/2102.10369)

## Reference

If you find the toolbox useful for your research, please cite the following:

```
@article{taog2023backdoorvault,
  title={{BackdoorVault}: A Toolbox for Backdoor Attacks},
  author={Tao, Guanhong and Cheng, Siyuan},
  year={2023}
}
```
