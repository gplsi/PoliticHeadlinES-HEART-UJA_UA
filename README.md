# HEART-UJA-UA at PoliticHeadlinES-IberLEF 2026

## Abstract

This paper presents our participation in PoliticHeadlinES 2026, a shared task on ranking candidate headlines for Spanish political news articles. We compare encoder-based and generative approaches, including direct semantic similarity, a lightweight pairwise neural ranker built on frozen multilingual embeddings, and a retrieval-augmented few-shot prompting pipeline. Our final results show that the generative approach achieves the best overall performance, indicating that in-context ranking with retrieved examples is particularly effective for this task. We also explore multimodal late fusion with CLIP-based image embeddings and different strategies for representing long articles, finding that visual information provides limited benefits and that truncating the article to the first 512 tokens remains a strong baseline for encoder models.


## Citation

<!-- ```bibtex
@inproceedings{moreno2026heart,
  title = {{HEART-UJA-UA} at {PoliticHeadlinES-IberLEF} 2026: Generative and Encoder-Based Approaches for Spanish Political Headline Ranking},
  author = {Moreno-Muñoz, Adrián and Canal-Esteve, Miquel and Martín-Valdivia, María-Teresa and Gutiérrez, Yoan and Ureña-López, Luis-Alfonso and Llopis, Fernando and Martínez-Cámara, Eugenio},
  booktitle = {Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2026)},
  year = {2026},
  publisher = {CEUR-WS.org},
}
```

```bibtex
@inproceedings{gomez2026politicheadlines,
  title = {Overview of {PoliticHeadlinES} at {IberLEF} 2026: Multimodal Headline Ranking in Spanish Political News},
  author = {Gómez-Navalón, J. and Bernal-Beltrán, T. and Pan, R. and García-Díaz, J. A. and Valencia-García, R.},
  booktitle = {Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2026)},
  year = {2026},
  publisher = {CEUR-WS.org},
}
``` -->

## Contact

For questions about this work, please contact:

- **Adrián Moreno-Muñoz** — [ammunoz@ujaen.es](mailto:ammunoz@ujaen.es)  
  CEATIC, University of Jaén, Spain

- **Miquel Canal-Esteve** — [mikel.canal@ua.es](mailto:mikel.canal@ua.es)  
  University of Alicante, Spain


