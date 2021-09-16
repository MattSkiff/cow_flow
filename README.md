This repository is directly based on the 'DifferNet' normalizing flow architecture by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.

Please see here:
https://github.com/marco-rudolph/differnet <br />
Paper: https://arxiv.org/pdf/2008.12577v1.pdf

The network has been modified to take in density maps instead of pictures of anomalies. These are conditioned on features extracted from the corresponding aerial images. The density maps were produced from the centre points of a manually curated dataset of 6,000 images, with approximately 400 of these having object annotations.

# License

This project is licensed under the MIT License.
