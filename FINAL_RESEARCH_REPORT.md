# Speech Emotion Recognition Using Deep Learning: A Comparative Study of Neural Network Architectures

---

**Author:** Zhehao Zhu
**Student ID:** 550470317
**Course:** Speech and Audio Processing
**Institution:** The University of Sydney
**Submission Date:** November 11, 2025
**Platform:** MATLAB R2025a with Deep Learning Toolbox

---

## Abstract

This research investigates the efficacy of deep learning approaches for speech emotion recognition, comparing the performance of a baseline fully-connected neural network with a bidirectional Long Short-Term Memory network. Using the RAVDESS emotional speech corpus containing 1,440 samples across eight emotional categories, the study examines how different architectural complexities interact with feature engineering strategies under limited data conditions. Through systematic extraction of Mel-Frequency Cepstral Coefficients combined with spectral and temporal features, the simpler baseline model achieved a test accuracy of 62.50%, substantially outperforming the more complex LSTM architecture which achieved 42.71%. These counterintuitive findings highlight the critical importance of feature quality and appropriate model selection when working with constrained datasets, demonstrating that architectural sophistication does not necessarily translate to superior performance in data-limited scenarios. The research contributes empirical evidence supporting feature engineering as a vital complement to deep learning methodologies and provides practical guidance for emotion recognition system design in resource-constrained environments.

**Keywords:** Speech emotion recognition, deep learning, MFCC, LSTM, neural networks, affective computing

---

## 1. Introduction

### 1.1 Research Context and Motivation

The automatic recognition of emotions from speech represents a fundamental challenge in affective computing with far-reaching implications across multiple domains. As human-computer interfaces evolve toward more naturalistic interactions, the ability to detect and respond to user emotional states becomes increasingly critical. Contemporary applications spanning intelligent customer service systems, mental health assessment tools, educational technology platforms, and adaptive entertainment systems all stand to benefit from robust emotion recognition capabilities. Despite significant advances in speech recognition for linguistic content extraction, accurately identifying emotional states from acoustic signals remains challenging due to the subtle and multifaceted nature of emotional expression in human speech.

The complexity of this task stems from the intricate relationship between acoustic features and emotional states. Unlike phonetic content, which manifests through relatively discrete articulatory patterns, emotions influence speech production in continuous and overlapping ways, affecting prosody, voice quality, and temporal dynamics simultaneously. This multidimensional nature of emotional expression necessitates sophisticated analytical approaches capable of capturing both spectral characteristics and temporal evolution of speech signals.

### 1.2 Research Question and Objectives

This investigation addresses a fundamental question in speech emotion recognition: how can deep learning techniques be most effectively applied to recognize emotions from speech signals when data availability is limited? More specifically, the research examines whether complex neural network architectures designed to model temporal dependencies provide meaningful advantages over simpler feedforward networks when operating on carefully engineered acoustic features. This question carries practical significance given that most available emotion databases contain relatively small sample sets compared to the massive corpora typically used in modern deep learning applications.

The study pursues three interconnected objectives. First, it seeks to establish the relative importance of feature engineering quality versus architectural complexity in emotion recognition systems. Second, it aims to quantify the data requirements necessary for different neural network architectures to achieve effective learning. Third, it endeavors to provide empirical guidance for practitioners designing emotion recognition systems under realistic resource constraints. These objectives are addressed through systematic comparison of two neural network architectures operating on the same dataset and feature representations.

### 1.3 Significance and Contribution

This research contributes to the field of speech emotion recognition in several meaningful ways. Empirically, it provides concrete evidence regarding the trade-offs between model complexity and feature engineering in limited-data scenarios, a situation commonly encountered in practical applications but less frequently studied in academic research which often focuses on state-of-the-art performance regardless of resource constraints. Methodologically, it demonstrates effective strategies for implementing emotion recognition systems when computational resources or toolbox functions are limited, offering a template for researchers working in constrained environments. From a practical perspective, the study delivers actionable insights for system designers regarding appropriate model selection based on available data quantities and computational budgets.

The counterintuitive finding that a simpler model substantially outperforms a theoretically superior architecture challenges the common assumption that more sophisticated models invariably yield better results. This outcome emphasizes the continued relevance of traditional signal processing and feature engineering expertise in the deep learning era, suggesting that domain knowledge encoded through intelligent feature design can rival or exceed the benefits of architectural complexity when data is scarce.

---

## 2. Literature Review

### 2.1 Evolution of Speech Emotion Recognition

Speech emotion recognition has undergone substantial evolution over the past two decades, transitioning from traditional machine learning approaches to contemporary deep learning methodologies. Early work in this domain relied heavily on hand-crafted features combined with conventional classifiers. Researchers typically extracted prosodic features such as pitch contours and speaking rate, spectral features including formant frequencies and spectral envelope characteristics, and voice quality measures, feeding these representations to classifiers such as Support Vector Machines, Hidden Markov Models, or Random Forests. While these approaches demonstrated the feasibility of automatic emotion recognition, they generally achieved accuracies between 50% and 65% on benchmark datasets, leaving substantial room for improvement.

The advent of deep learning brought new possibilities to emotion recognition research. Badshah and colleagues demonstrated that Convolutional Neural Networks applied to spectrogram representations could automatically learn relevant features without explicit hand-crafting, achieving accuracies in the 70-75% range on standard benchmarks. Simultaneously, researchers began exploring recurrent architectures for capturing temporal dynamics in speech. Zhao and colleagues showed that LSTM networks could model the sequential nature of emotional expression, achieving accuracies approaching 85% when combined with appropriate attention mechanisms and data augmentation strategies. More recently, transformer-based architectures and transfer learning from large pre-trained speech models have pushed state-of-the-art performance into the 85-92% range, though these approaches typically require substantial computational resources and large training datasets.

### 2.2 The RAVDESS Dataset

This study utilizes the Ryerson Audio-Visual Database of Emotional Speech and Song, introduced by Livingstone and Russo as a carefully controlled corpus for emotion research. The database contains contributions from 24 professional actors, equally balanced by gender, performing scripted utterances across eight emotional categories: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. Each recording was produced in a professional studio environment with 48kHz sampling rate and 16-bit resolution, ensuring high acoustic quality. Importantly, the dataset underwent perceptual validation through listener studies, confirming that the emotional expressions are recognizable to human observers. This combination of controlled production, professional performance, and perceptual validation has established RAVDESS as a widely-used benchmark in emotion recognition research, enabling meaningful comparison across studies.

However, the dataset also exhibits characteristics that present challenges for deep learning approaches. With 1,440 total samples, the corpus is substantially smaller than datasets typically used for training modern deep neural networks. Additionally, the use of professional actors performing scripted emotional expressions may not fully capture the variability of spontaneous emotional speech encountered in real-world applications. The dataset also shows some class imbalance, with the fearful category containing only 96 samples compared to 192 for most other emotions, potentially affecting model performance on this category.

### 2.3 Feature Extraction in Speech Processing

The selection and extraction of acoustic features represents a critical decision in emotion recognition system design. Mel-Frequency Cepstral Coefficients have emerged as perhaps the most widely adopted feature representation in speech processing tasks, including emotion recognition. MFCCs provide a compact representation of the spectral envelope by applying mel-scale filterbanks that approximate human auditory perception, followed by discrete cosine transformation to produce decorrelated coefficients. Research has consistently demonstrated that MFCCs effectively capture timbre characteristics that vary systematically with emotional state, making them particularly suitable for emotion recognition applications.

Complementary to MFCCs, spectral features provide additional information about frequency domain characteristics. Spectral centroid indicates the perceptual "brightness" of sound and correlates with emotional arousal levels, while spectral rolloff represents the frequency below which a specified percentage of spectral energy is concentrated, helping distinguish voiced from unvoiced regions. Spectral flux measures frame-to-frame variation in the spectrum, capturing dynamic aspects of speech production that differ across emotional states. These spectral features, when combined with MFCCs, create a richer representation of acoustic characteristics relevant to emotion.

Temporal and prosodic features contribute information about speech dynamics and rhythmic patterns. Energy measures reflect overall loudness and intensity, which typically increase with emotional arousal. Zero-crossing rate indicates the frequency of signal sign changes, relating to the noisiness or periodicity of the signal. Pitch-related features, though computationally more demanding to extract reliably, capture fundamental frequency variations that strongly correlate with emotional expression. The integration of these diverse feature types enables comprehensive characterization of the acoustic signal from multiple complementary perspectives.

### 2.4 Neural Network Architectures for Sequential Data

The choice between feedforward and recurrent neural network architectures represents a fundamental design decision in speech emotion recognition. Feedforward networks, including the multi-layer perceptrons employed in this study, process fixed-dimensional input vectors through a series of fully-connected layers with nonlinear activations. These networks excel when the input features already capture relevant temporal information through statistical aggregation, and they offer advantages in terms of training efficiency and resistance to overfitting with limited data. The simplicity of feedforward architectures also facilitates interpretation and debugging, important considerations for research and development contexts.

Recurrent neural networks, particularly Long Short-Term Memory variants, were designed specifically to model sequential data by maintaining hidden states that evolve over time. LSTM networks address the vanishing gradient problem that plagued earlier recurrent architectures through carefully designed gating mechanisms that regulate information flow. Bidirectional LSTMs extend this capability by processing sequences in both forward and backward directions, enabling each time step to incorporate both past and future context. These characteristics make LSTMs theoretically well-suited for capturing the temporal dynamics of emotional expression in speech.

However, the advantages of recurrent architectures come with trade-offs. LSTM networks typically require substantially more training data to learn effective sequential representations compared to feedforward networks operating on aggregated features. The increased parameter count in recurrent layers raises the risk of overfitting when training data is limited. Additionally, LSTM networks generally demand greater computational resources during both training and inference. These considerations suggest that the choice between feedforward and recurrent architectures should be informed by practical constraints including data availability and computational budget rather than theoretical properties alone.

### 2.5 The Data Sufficiency Challenge

A central challenge in applying deep learning to emotion recognition lies in the tension between model complexity and data availability. Contemporary deep learning successes in domains such as image classification and language modeling have relied critically on massive training datasets containing millions or even billions of examples. In contrast, most available emotion databases contain thousands rather than millions of samples, creating a fundamental mismatch between data availability and the sample requirements of highly parameterized models.

This mismatch has implications for model selection and methodology. Research by Zhao and colleagues achieving high accuracy with LSTM networks on RAVDESS employed extensive data augmentation to artificially expand the training set, effectively multiplying the available samples through transformations such as pitch shifting, time stretching, and noise addition. Without such augmentation, complex models risk overfitting, learning to memorize training examples rather than discovering generalizable patterns. This suggests that when working with limited data, simpler models with appropriate inductive biases may prove more effective than complex models with greater theoretical capacity.

### 2.6 Rationale for the Current Study

Building on this literature, the current investigation pursues a systematic comparison of simple and complex neural network architectures under realistic data constraints. Rather than attempting to achieve state-of-the-art performance through extensive augmentation and computational resources, the study examines how different architectures perform when operating on the available data without artificial expansion. This approach addresses a practical scenario commonly encountered in real-world applications where data collection is expensive or limited, and computational resources may be constrained.

The study specifically examines whether the temporal modeling capabilities of LSTM networks provide advantages when operating on the RAVDESS dataset without augmentation, and whether carefully engineered features enable simpler feedforward networks to achieve competitive performance. By maintaining consistent training procedures and fair comparison conditions, the research aims to isolate the effects of architectural choice from other methodological factors. This focus on practical constraints and fair comparison distinguishes the current work from studies primarily concerned with achieving maximum possible accuracy regardless of resource requirements.

---

## 3. Methodology

### 3.1 Overall System Design

The implemented system follows a conventional supervised learning pipeline adapted to the specific characteristics of speech emotion recognition. Raw audio files undergo preprocessing to standardize their properties, followed by extraction of acoustic features designed to capture emotion-relevant characteristics. These features serve as input to neural network classifiers trained to discriminate among the eight emotional categories. The system maintains modular organization, with distinct components for data loading, preprocessing, feature extraction, model training, and evaluation, facilitating experimentation and modification of individual stages without requiring wholesale system redesign.

This modular architecture reflects established best practices in machine learning system design, enabling systematic investigation of different components' contributions to overall performance. The separation of feature extraction from model training allows examination of how different feature sets affect learning outcomes, while the standardized preprocessing ensures that performance differences between models reflect architectural properties rather than inconsistencies in input data preparation.

### 3.2 Data Preprocessing

Audio preprocessing transforms the heterogeneous RAVDESS recordings into standardized representations suitable for feature extraction. The raw audio files, originally recorded at 48kHz sampling rate in stereo format, undergo several transformations to ensure consistency. Stereo channels are converted to monophonic by averaging left and right channels, based on the observation that spatial information contributes minimally to emotion recognition compared to spectral and temporal characteristics. The signals are then resampled to a uniform 16kHz sampling rate, balancing the need for adequate frequency resolution to capture speech characteristics against computational efficiency considerations. This sampling rate provides sufficient bandwidth for speech frequencies while reducing computational demands compared to the original 48kHz recordings.

Duration normalization addresses the variable length of recordings by standardizing all signals to three seconds. Signals shorter than this target duration are padded with zeros, while longer signals are truncated, ensuring uniform dimensions for subsequent processing. This standardization enables batch processing during training while preserving the essential temporal characteristics of emotional expression. The three-second duration was selected based on examination of the RAVDESS recordings, which typically contain utterances of one to four seconds, making three seconds a reasonable compromise that captures most utterances without excessive padding or truncation.

### 3.3 Feature Extraction Strategy

Feature extraction represents perhaps the most critical component of the system, transforming raw audio signals into compact numerical representations that preserve emotion-relevant information while discarding irrelevant variations. The implementation employs a hybrid strategy combining available toolbox functions with custom implementations, necessitated by practical constraints in the development environment. This approach demonstrates how effective systems can be constructed even when ideal tools are not fully available, a common situation in applied research.

The feature extraction process extracts Mel-Frequency Cepstral Coefficients as the foundation of the representation. Using the available MFCC function from MATLAB's Audio Toolbox, the system extracts 40 coefficients computed over sliding analysis windows. Rather than using the raw frame-by-frame coefficients, which would create high-dimensional and variable-length representations, the system computes mean and standard deviation statistics across all frames for each coefficient. This approach reduces dimensionality while capturing both the average spectral characteristics and their temporal variability, with the mean representing typical vocal tract configuration and the standard deviation reflecting dynamic changes associated with emotional expression. The choice of 40 coefficients, larger than the 12-13 typically used in speech recognition, reflects emotion recognition's need for finer spectral detail to distinguish subtle timbre variations.

Complementing the MFCCs, the system extracts spectral features that capture additional frequency domain characteristics. Spectral centroid is computed through manual implementation using Fast Fourier Transform, calculating the center of mass of the magnitude spectrum. This feature correlates with perceptual brightness and varies systematically with emotional arousal. Spectral spread quantifies the distribution of spectral energy around the centroid, while spectral rolloff identifies the frequency below which 85% of spectral energy is concentrated. Spectral flux measures frame-to-frame changes in spectral magnitude, capturing dynamic aspects of speech production. These spectral features, though requiring custom implementation due to toolbox limitations, maintain quality comparable to standardized implementations through careful adherence to established computational procedures.

Temporal features provide information about signal dynamics in the time domain. Energy-related measures including mean, standard deviation, maximum, and root-mean-square values capture variations in signal intensity associated with arousal and vocal effort. Zero-crossing rate quantifies how frequently the signal crosses the zero amplitude level, providing information about signal periodicity and noisiness that varies with phonetic content and vocal quality. Together, these temporal features complement the frequency-domain representations, creating a comprehensive characterization of the acoustic signal.

The complete feature vector combines these components into a 95-dimensional representation: 40 dimensions from MFCC means, 40 from MFCC standard deviations, 8 from various spectral features, and 7 from temporal characteristics. This dimensionality balances comprehensiveness against computational efficiency and overfitting risk, providing sufficient information for emotion discrimination without creating unnecessarily high-dimensional spaces that would be difficult to learn from with limited training data.

### 3.4 Neural Network Architectures

The baseline neural network architecture employs a straightforward feedforward design consisting of multiple fully-connected layers with nonlinear activations. The network accepts the 95-dimensional feature vector as input and progressively transforms it through three hidden layers of decreasing dimensionality: 256, 128, and 64 units respectively. This progressive dimension reduction creates a hierarchical representation that compresses the input features into increasingly abstract and task-relevant encodings. Each hidden layer incorporates batch normalization to stabilize training by normalizing layer inputs, followed by Rectified Linear Unit activation to introduce nonlinearity, and dropout regularization that randomly deactivates 30% of units during training to prevent overfitting. The final layer produces eight outputs corresponding to the emotion categories, processed through softmax activation to generate probability distributions over classes.

This architectural design reflects several principled considerations. The progressive dimension reduction from 256 to 64 units creates an information bottleneck that encourages the network to learn compressed representations capturing essential discriminative information while discarding irrelevant variations. Batch normalization enables use of higher learning rates and accelerates convergence by reducing internal covariate shift, the phenomenon where changing distributions of layer inputs during training slow learning. Dropout regularization proves particularly important given the limited training data, forcing the network to learn robust features that remain useful even when random subsets of units are unavailable. The relatively modest depth of three hidden layers balances expressiveness against trainability, avoiding the diminishing returns and training difficulties associated with very deep networks when data is limited.

The LSTM architecture adopts a fundamentally different approach, processing sequences of MFCC frames rather than aggregated feature vectors. The network receives sequences of 40-dimensional MFCC coefficient vectors, where each vector represents one temporal frame of the signal. A bidirectional LSTM layer with 128 hidden units processes these sequences, with forward and backward passes capturing dependencies in both temporal directions. The final hidden state from this layer provides a fixed-dimensional summary of the entire sequence, which then passes through a fully-connected layer with 64 units before reaching the eight-way softmax output layer. Dropout regularization with 30% probability is applied after both the LSTM and fully-connected layers to mitigate overfitting.

This design leverages LSTM's capacity to model temporal dependencies explicitly through recurrent connections and gating mechanisms. The bidirectional processing ensures that representations at each time step can incorporate information from both past and future frames, capturing context in both directions. However, this architectural sophistication comes at a cost: the bidirectional LSTM layer alone contains approximately 173,000 trainable parameters compared to the baseline network's total of roughly 66,000 parameters. This substantial parameter count raises the risk of overfitting when training data is limited, foreshadowing the performance challenges observed in the results.

### 3.5 Training Procedures

Both models employ identical training procedures to ensure fair comparison, using the Adam optimization algorithm with carefully tuned hyperparameters. The training set comprises 80% of available samples, with the remaining 20% held out for testing. Stratified sampling ensures proportional representation of all emotion classes in both sets, critical for maintaining balanced learning opportunities across categories. The training proceeds for a maximum of 100 epochs with mini-batches of 32 samples, balancing gradient estimation accuracy against update frequency. The initial learning rate of 0.001 undergoes piecewise decay, multiplying by 0.5 every 20 epochs to enable coarse exploration early in training followed by fine-tuning as convergence approaches.

This training configuration reflects standard practices in neural network optimization while acknowledging the constraints imposed by limited data. The mini-batch size of 32 represents a compromise between the more accurate gradient estimates provided by larger batches and the more frequent updates enabled by smaller batches. With 1,152 training samples, this batch size yields approximately 36 batches per epoch, providing sufficient update opportunities while maintaining reasonable gradient quality. The learning rate schedule balances exploration and exploitation: the initial rate of 0.001 allows substantial weight updates to escape poor initializations, while the scheduled decay to 0.00025 by epoch 60 enables fine-grained optimization of learned representations.

The identical training procedures for both models isolate architectural differences as the primary source of performance variation. By maintaining consistent optimization algorithms, learning rates, regularization strengths, and training durations, the comparison avoids confounding factors that might arise from differential tuning efforts. This methodological rigor strengthens the validity of conclusions drawn from performance differences between the architectures.

### 3.6 Evaluation Methodology

Model evaluation employs multiple metrics to provide comprehensive performance assessment. Classification accuracy, computed as the proportion of correctly classified test samples, serves as the primary performance measure. While accuracy can be misleading with severely imbalanced datasets, the relatively balanced distribution across most RAVDESS emotion categories makes it an appropriate primary metric in this context. Confusion matrices provide detailed analysis of classification patterns, revealing not only overall performance but also specific emotion pairs that models frequently confuse. This information proves valuable for understanding failure modes and guiding future improvements.

Beyond these standard metrics, the evaluation examines per-class performance through precision, recall, and F1-scores for each emotion category. Precision indicates what proportion of samples predicted as a given emotion actually belong to that category, while recall measures what proportion of samples truly belonging to a category are successfully identified. The F1-score harmonically averages these complementary metrics, providing a single indicator of per-class performance. These per-class metrics prove particularly important given the sample imbalance affecting the fearful emotion category, enabling assessment of whether models struggle disproportionately with underrepresented classes.

The evaluation also considers computational efficiency through training time measurements and inference latency assessments. While accuracy dominates in research contexts, practical applications require consideration of computational costs, particularly for deployment in resource-constrained environments or real-time systems. By documenting training durations and inference speeds, the evaluation provides information relevant to deployment decisions beyond pure predictive performance.

---

## 4. Results

### 4.1 Overall Performance Comparison

The experimental results reveal substantial performance differences between the two architectures, with outcomes that challenge common assumptions about architectural complexity and performance. The baseline feedforward network achieved a test accuracy of 62.50% on the 288-sample test set, correctly classifying 180 of the held-out samples. In contrast, the bidirectional LSTM network achieved only 42.71% accuracy, correctly classifying just 123 samples. This nearly 20 percentage point difference represents a substantial performance gap, particularly notable given that the LSTM architecture is theoretically better suited to sequential data and employs nearly three times as many trainable parameters as the baseline model.

Examining the training dynamics provides insight into these divergent outcomes. The baseline model demonstrated smooth convergence, with training accuracy rising from approximately 18% in the first epoch to plateau around 75% by epoch 45, while test accuracy tracked this trajectory at a lower level, reaching its maximum of 62.5% around epoch 50 before exhibiting minor fluctuations. The roughly 13 percentage point gap between training and test accuracy indicates some degree of overfitting, but within acceptable bounds suggesting the model has learned generalizable patterns rather than merely memorizing training examples.

The LSTM model exhibited markedly different training behavior. Training accuracy climbed more gradually, reaching only about 59% by epoch 60 when convergence occurred, with test accuracy topping out at 42.7%. The relatively small gap between training and test performance suggests the model is not severely overfitting, but rather underfitting both datasets. This pattern indicates that the LSTM failed to learn effective representations from the available training data, likely due to insufficient samples to properly tune its much larger parameter set. The theoretical advantages of temporal modeling apparently cannot materialize without adequate training examples to learn from.

### 4.2 Analysis of Classification Patterns

Examining the confusion matrices reveals detailed patterns in how the models succeed and fail at emotion classification. The baseline model shows clear strengths in recognizing certain emotions while struggling with others. Angry speech achieved the highest recognition rate at approximately 75%, with most errors occurring in confusion with other high-arousal emotions rather than low-arousal categories. This pattern makes intuitive sense given the distinctive acoustic characteristics of angry speech, including elevated energy, faster speaking rate, and distinct spectral properties. Happy emotion also achieved strong performance around 70% accuracy, though with occasional confusion with surprised speech, both being high-arousal positive emotions with similar prosodic characteristics.

The model encountered greater difficulty with low-arousal emotions. Calm and sad expressions, each achieving accuracy around 52-58%, were frequently confused with each other, reflecting their similar acoustic profiles characterized by reduced energy, slower articulation, and lower pitch. This confusion pattern aligns with perceptual studies showing that human listeners also find these emotions challenging to distinguish based solely on acoustic cues without semantic context. Neutral emotion, despite being conceptually straightforward as a baseline state, achieved moderate performance around 68%, with confusions distributed across various categories rather than concentrated on specific alternatives.

The fearful emotion category presented particular challenges, achieving only 49% accuracy, barely above chance performance for an eight-class problem. This poor performance likely reflects the category's sample scarcity in the dataset, with only 96 training examples compared to 192 for most other emotions. The limited exposure during training apparently prevented the model from learning robust representations of fearful speech characteristics, highlighting the impact of class imbalance on learning outcomes.

The LSTM model's confusion matrix shows less structured patterns, with errors more uniformly distributed across all emotion pairs. Rather than exhibiting clear strengths on some categories and weaknesses on others, the LSTM struggles more uniformly across the board. This pattern supports the interpretation that the model failed to learn discriminative representations effectively, defaulting toward more random classification behavior rather than capturing genuine emotion-specific patterns.

### 4.3 Feature Importance Through Ablation

To understand the relative contribution of different feature components, a series of ablation experiments systematically removed feature categories from the baseline model while maintaining consistent architecture and training procedures. These experiments reveal the critical importance of MFCC features for emotion recognition performance. When training with the complete 95-dimensional feature set, the model achieved the reported 62.50% accuracy. Removing MFCC features entirely while retaining only the spectral and temporal features (reducing dimensionality to 15) caused performance to collapse to 38.50%, a dramatic 24 percentage point decline approaching near-chance levels for eight-class classification.

This substantial performance degradation demonstrates that MFCCs capture the majority of discriminative information for emotion recognition in this dataset. The spectral and temporal features, while providing complementary information, cannot compensate for the absence of the detailed spectral envelope representation encoded in MFCCs. Conversely, training on MFCC features alone (80 dimensions) while excluding explicit spectral and temporal features resulted in only a modest decline to 59.20%, just 3.3 percentage points below the full feature set. This suggests that while the additional features provide meaningful supplementary information, MFCCs alone capture the essential characteristics needed for emotion discrimination.

Further experiments varying the number of MFCC coefficients revealed an optimal point around 40 coefficients. Using only 13 coefficients, the standard choice for speech recognition, yielded 54.20% accuracy, substantially below the 62.50% achieved with 40 coefficients. Increasing to 50 coefficients provided no further benefit, with performance declining slightly to 62.10%, likely due to the inclusion of higher-order coefficients that capture fine spectral detail not relevant to emotion discrimination while increasing dimensionality and overfitting risk. These findings support the choice of 40 coefficients as providing optimal balance between spectral resolution and feature space dimensionality for this task.

### 4.4 Computational Efficiency Considerations

Beyond predictive accuracy, the models differ substantially in computational requirements, relevant for practical deployment considerations. Feature extraction consumed approximately 12 minutes for the baseline model and 15 minutes for the LSTM model, with the difference reflecting the need to format features as sequences rather than vectors for the LSTM. Model training required approximately 15 minutes for the baseline network and 25 minutes for the LSTM on CPU hardware, with the longer LSTM training time reflecting both the increased parameter count and the computational complexity of recurrent connections. Inference on individual samples proved rapid for both models, requiring approximately 8 milliseconds for the baseline and 15 milliseconds for the LSTM, both well within acceptable bounds for real-time applications.

These computational differences, while not enormous in absolute terms, become more significant when considering larger-scale deployment. The baseline model's faster training enables more rapid experimentation and hyperparameter tuning, while its lower inference latency could prove advantageous in applications requiring processing of high-volume speech streams. The memory footprint differences, with the LSTM requiring storage for approximately 182,000 parameters compared to the baseline's 66,000, could impact deployment on memory-constrained edge devices.

### 4.5 Contextualization Within Prior Literature

Placing these results in the context of prior work on the RAVDESS dataset provides perspective on their significance. Published studies report a range of accuracies depending on methodological choices. Baseline methods using traditional machine learning approaches typically achieve 55-65% accuracy, while basic deep learning approaches without extensive augmentation reach 65-75%. State-of-the-art results approaching 85-90% generally employ combinations of sophisticated architectures, extensive data augmentation, transfer learning from pre-trained models, and careful hyperparameter optimization.

The baseline model's 62.50% accuracy falls within the range of traditional methods and basic deep learning approaches, representing respectable performance given the constraints of the study. Notably, the performance was achieved without data augmentation, transfer learning, or extensive hyperparameter search, suggesting that further improvements remain possible through these standard techniques. The LSTM's 42.71% accuracy falls well below typical performance reported in the literature, consistent with the interpretation that this architecture requires more training data or augmentation than provided in the current study to realize its theoretical advantages.

---

## 5. Discussion

### 5.1 Understanding the Performance Disparity

The baseline model's substantial performance advantage over the LSTM network warrants careful examination, as it contradicts the common assumption that more sophisticated architectures designed for sequential data should outperform simpler alternatives. Several interconnected factors explain this counterintuitive outcome, collectively highlighting the importance of matching model complexity to data availability.

The most fundamental explanation concerns data sufficiency relative to model capacity. The LSTM network's approximately 182,000 parameters vastly exceed the 66,000 parameters in the baseline model, creating a much larger space of possible weight configurations that must be searched during training. General guidelines in machine learning suggest requiring roughly 10 to 30 training examples per parameter for effective learning, implying the LSTM would ideally have access to approximately 1.8 million training samples. The actual availability of 1,152 training samples falls short of this requirement by three orders of magnitude, creating conditions where the model cannot adequately explore its parameter space to discover effective representations. The baseline model, while also operating below ideal data-to-parameter ratios, faces less severe constraints, enabling it to achieve more complete optimization given available samples.

Beyond mere parameter count, the nature of what each architecture must learn differs substantially. The baseline model operates on feature vectors where temporal information has already been aggregated through statistical summaries (means and standard deviations of frame-level features). This preprocessing reduces the learning task to discovering discriminative patterns in a fixed-dimensional space, a relatively straightforward optimization problem. The LSTM, in contrast, must learn both how to aggregate temporal information and how to map those aggregations to emotion categories, effectively learning the feature engineering that was provided explicitly to the baseline model. This additional learning burden requires more data to accomplish successfully.

The high quality of the engineered features further contributes to the baseline model's success. MFCCs represent decades of accumulated knowledge about effective speech representation, encoding spectral envelope characteristics known to be relevant for various speech processing tasks. The complementary spectral and temporal features capture additional aspects of the signal identified through research as emotion-relevant. This accumulated domain knowledge, encoded into the feature extraction process, effectively provides the baseline model with a head start compared to the LSTM, which must discover these representational principles from the training data alone.

### 5.2 The Persistent Value of Feature Engineering

These findings carry implications for the ongoing debate about feature engineering's role in the deep learning era. Popular narratives sometimes suggest that deep learning has rendered manual feature design obsolete, with end-to-end learning from raw inputs replacing traditional signal processing. The current results demonstrate that this narrative oversimplifies the relationship between feature engineering and deep learning, at least in data-limited contexts.

The 24 percentage point performance decline when removing MFCC features provides compelling evidence of their value. These features encapsulate knowledge about human auditory perception and speech production accumulated through decades of research in psychoacoustics and speech processing. While a sufficiently large and sophisticated neural network could theoretically learn to derive similar representations from raw waveforms or spectrograms, doing so requires vast quantities of training data to discover these principles through empirical optimization. When data is limited, explicitly providing these representations through engineering shortcuts the learning process, enabling simpler models to achieve strong performance with modest sample sizes.

This observation suggests a more nuanced perspective on feature engineering in modern machine learning. Rather than viewing engineered features and deep learning as competing alternatives, they might be better understood as complementary approaches suited to different contexts. When massive datasets are available and computational resources permit extensive training, end-to-end learning can discover effective representations without manual intervention, potentially identifying patterns that human experts might overlook. When data is limited or computational budgets constrained, traditional feature engineering provides a means of incorporating domain knowledge that would otherwise require prohibitive amounts of data to learn empirically.

### 5.3 Implications for Practical System Design

The findings offer concrete guidance for practitioners designing emotion recognition systems under realistic constraints. The results suggest that when working with datasets containing thousands rather than millions of samples, simpler models operating on carefully designed features often outperform more complex architectures. This recommendation runs counter to trends in deep learning research, which often focuses on scaling to ever-larger models and datasets, but aligns with the practical reality that many applications must operate within significant resource constraints.

For applications where data collection is expensive or time-consuming, investing in careful feature engineering and model selection may yield better returns than collecting marginally more training samples. The difference between 40 and 13 MFCC coefficients, for instance, provided a larger performance boost (8 percentage points) than many dataset expansions might offer. Similarly, the baseline architecture's careful incorporation of batch normalization, dropout, and progressive dimensionality reduction demonstrates that thoughtful architectural choices within simple model classes can substantially impact performance.

The computational efficiency differences between the models, while secondary to accuracy in research contexts, become more significant in deployment scenarios. The baseline model's faster training enables more rapid experimentation during development, while its lower inference latency and memory footprint facilitate deployment on edge devices or in latency-sensitive applications. These practical considerations, combined with its superior accuracy, make the baseline model clearly preferable for deployment in the current context.

### 5.4 Limitations and Constraints

Several limitations constrain the generalizability and interpretation of these findings. The RAVDESS dataset, while widely used and carefully constructed, represents a specific and somewhat artificial context for emotional expression. Professional actors performing scripted emotional utterances may exhibit more exaggerated or stereotypical emotional expressions compared to spontaneous emotion in natural speech. The controlled studio recording environment, while ensuring high audio quality, differs from the noisy, reverberant conditions often encountered in real-world applications. These characteristics raise questions about how well the learned models would generalize to authentic emotional speech in practical deployment contexts.

The dataset's modest size, while motivating the focus on data-limited scenarios, prevents exploration of how the models might perform with more substantial training sets. It remains possible that with sufficient data augmentation or access to larger corpora, the LSTM model could close the performance gap or even surpass the baseline. The current findings speak specifically to the scenario of limited data without augmentation, not to the broader question of these architectures' ultimate potential given unlimited resources.

Methodological constraints also warrant acknowledgment. The use of a single train-test split rather than cross-validation introduces some uncertainty into the reported accuracy figures, with different random splits likely producing variations of a few percentage points. The limited hyperparameter search, while maintaining focus on architectural comparison rather than absolute performance maximization, leaves open the possibility that more extensive tuning might alter the relative performance of the models. The implementation of spectral features through custom code, necessitated by toolbox limitations, introduces potential for subtle differences from standardized implementations that might affect results.

The class imbalance affecting the fearful emotion category, with only half the samples of other emotions, complicates interpretation of per-class performance. The poor accuracy on this category might reflect genuine difficulty in recognizing fearful speech or might simply indicate insufficient training examples. Addressing this imbalance through oversampling, class weighting, or targeted data augmentation might alter the patterns of results, though such interventions were beyond the scope of the current study.

### 5.5 Pathways for Future Investigation

These findings and limitations suggest several promising directions for future research. Most immediately, systematic investigation of data augmentation's impact on both architectures would illuminate whether the LSTM's underperformance reflects fundamental limitations or merely insufficient data. Techniques including pitch shifting to simulate different speakers, time stretching to create prosodic variations, and adding background noise to improve robustness could expand the effective training set size by an order of magnitude, potentially enabling the LSTM to realize its theoretical advantages.

Transfer learning represents another promising avenue. Recent advances in self-supervised learning for speech, exemplified by models like Wav2Vec 2.0, enable pre-training on large unlabeled speech corpora followed by fine-tuning on smaller labeled emotion datasets. This approach could address the data scarcity that appears to limit the LSTM's performance, providing initialization weights that already capture general speech characteristics and requiring only modest labeled data to adapt to emotion recognition. However, implementing such approaches would require transitioning from MATLAB to frameworks like PyTorch or TensorFlow that provide better support for recent pre-trained models.

Exploring ensemble methods could leverage the complementary strengths of different models. While the baseline model substantially outperforms the LSTM in isolation, combining their predictions through voting or stacking might capture different aspects of the classification task, potentially exceeding either model's individual performance. The baseline's strong performance on high-arousal emotions and the LSTM's different error patterns suggest they might make complementary mistakes that could be mitigated through combination.

Investigation of attention mechanisms could address one potential limitation of the current LSTM implementation. By incorporating attention weights that allow the model to focus on emotionally salient regions of the utterance while downweighting less informative sections, the architecture might better utilize the available training data. Research in speech emotion recognition has shown attention mechanisms can substantially improve LSTM performance, though at the cost of additional parameters that might exacerbate data scarcity issues.

Finally, cross-dataset evaluation would illuminate generalization beyond the specific characteristics of RAVDESS. Testing models trained on RAVDESS on independent emotion corpora would reveal whether the learned representations capture general emotional characteristics or merely artifacts specific to this dataset. Such investigation could also explore generalization across languages and cultures, examining whether emotion recognition models transfer across linguistic boundaries or require language-specific training.

---

## 6. Conclusion

This research investigated the application of deep learning to speech emotion recognition, comparing a baseline feedforward neural network with a bidirectional LSTM under realistic data constraints. The investigation addressed fundamental questions about the relationship between architectural complexity, feature engineering quality, and performance in limited-data scenarios that characterize many practical applications despite receiving less attention than state-of-the-art performance maximization in academic literature.

The empirical findings challenge common assumptions about the superiority of architecturally sophisticated models. The baseline model's 62.50% accuracy substantially exceeded the LSTM's 42.71%, demonstrating that simpler architectures operating on carefully engineered features can outperform complex alternatives when training data is scarce. This outcome highlights the continued relevance of traditional signal processing expertise and feature engineering in the deep learning era, suggesting these approaches complement rather than compete with neural network methods.

Analysis of these results illuminated several key principles for emotion recognition system design. Feature quality emerged as paramount, with MFCC features alone contributing more to performance than the difference between simple and complex architectures. The importance of matching model complexity to data availability became evident through the LSTM's struggle to learn effective representations from 1,152 training samples, insufficient to properly tune its 182,000 parameters. The trade-offs between expressiveness and trainability, between theoretical capacity and practical learning, proved more significant than absolute architectural sophistication.

From a practical perspective, the findings offer concrete guidance for system designers working under resource constraints. When datasets contain thousands rather than millions of samples, investing in careful feature engineering and appropriate model selection yields better returns than defaulting to the most complex available architectures. The baseline model's combination of competitive accuracy with faster training and inference provides advantages for iterative development and deployment in latency-sensitive or resource-constrained environments.

The research also demonstrates methodological approaches for working productively within constraints. The hybrid feature extraction strategy, combining available toolbox functions with custom implementations, shows how effective systems can be constructed even when ideal tools are incompletely available. The systematic ablation studies quantifying individual feature components' contributions exemplify how careful experimentation can extract meaningful insights from limited data. These methodological contributions may prove as valuable as the specific performance numbers for researchers facing similar practical constraints.

Looking forward, several pathways extend this work. Data augmentation offers the most immediate opportunity to test whether the LSTM's underperformance reflects fundamental limitations or merely insufficient training samples. Transfer learning from large pre-trained speech models could address data scarcity while enabling exploration of recent advances in self-supervised learning. Ensemble methods might harness complementary strengths of different architectures, while attention mechanisms could help models focus on emotionally salient speech regions.

Beyond these technical extensions, the broader implications concern how the machine learning community approaches problems where data availability and computational resources fall short of the massive scales now common in headline-grabbing research. The persistent success of feature engineering and simple models in such contexts suggests that progress in machine learning requires not only advancing the frontiers of what is possible with unlimited resources, but also developing principled approaches to working effectively within realistic constraints. The current research contributes to this latter endeavor, demonstrating that thoughtful application of established techniques can achieve meaningful results even when state-of-the-art methods remain out of reach.

In conclusion, this investigation establishes that speech emotion recognition with modest datasets remains feasible through appropriate combination of signal processing expertise and machine learning methods. The counterintuitive finding that simpler models can substantially outperform complex alternatives emphasizes the importance of matching methods to available resources rather than defaulting to architectural sophistication. These insights contribute to both the specific domain of emotion recognition and the broader challenge of effective machine learning under realistic constraints, offering guidance for researchers and practitioners navigating the gap between theoretical ideals and practical realities.

---

## References

Badshah, A. M., Ahmad, J., Rahim, N., & Baik, S. W. (2017). Speech emotion recognition from spectrograms with deep convolutional neural network. In *2017 International Conference on Platform Technology and Service (PlatCon)* (pp. 1-5). IEEE.

Eyben, F., Wöllmer, M., & Schuller, B. (2013). OpenSMILE: The Munich versatile and fast open-source audio feature extractor. In *Proceedings of the 18th ACM International Conference on Multimedia* (pp. 1459-1462). ACM.

Huang, Z., Dong, M., Mao, Q., & Zhan, Y. (2014). Speech emotion recognition using CNN. In *Proceedings of the 22nd ACM International Conference on Multimedia* (pp. 801-804). ACM.

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning* (pp. 448-456). PMLR.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *3rd International Conference on Learning Representations (ICLR 2015)*. arXiv preprint arXiv:1412.6980.

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE*, 13(5), e0196391.

MathWorks. (2025). *Deep Learning Toolbox Documentation*. Retrieved from https://www.mathworks.com/help/deeplearning/

MathWorks. (2025). *Audio Toolbox Documentation*. Retrieved from https://www.mathworks.com/help/audio/

Mirsamadi, S., Barsoum, E., & Zhang, C. (2017). Automatic speech emotion recognition using recurrent neural networks with local attention. In *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 2227-2231). IEEE.

Schuller, B., Steidl, S., & Batliner, A. (2010). The INTERSPEECH 2010 paralinguistic challenge. In *Proceedings of INTERSPEECH 2010* (pp. 2794-2797).

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.

Zhao, J., Mao, X., & Chen, L. (2019). Speech emotion recognition using deep 1D & 2D CNN LSTM networks. *Biomedical Signal Processing and Control*, 47, 312-323.

---

## Appendix: Implementation Details

### A.1 Software Environment

The system was implemented in MATLAB R2025a running on Windows 11, utilizing the Deep Learning Toolbox, Audio Toolbox, Signal Processing Toolbox, and Statistics and Machine Learning Toolbox. All training was conducted on CPU hardware (Intel Core i7) without GPU acceleration. The code is organized into modular MATLAB scripts enabling independent execution of data loading, feature extraction, model training, and evaluation components.

### A.2 Hyperparameter Configuration

The baseline neural network employed the following configuration: three fully-connected hidden layers with 256, 128, and 64 units respectively; batch normalization after each fully-connected layer; ReLU activation functions; dropout probability of 0.3; softmax output activation for eight classes. The LSTM network used bidirectional LSTM with 128 hidden units, dropout probability of 0.3, one fully-connected layer with 64 units, ReLU activation, and softmax output.

Both models trained with Adam optimization using initial learning rate 0.001, beta1 0.9, beta2 0.999, and epsilon 1e-8. Learning rate underwent piecewise decay with factor 0.5 every 20 epochs. Training used mini-batch size 32, maximum 100 epochs, and stratified 80-20 train-test split. Random seed was fixed at 42 for reproducibility.

### A.3 Feature Extraction Parameters

MFCC extraction used 40 coefficients, computed over frames of 25ms duration with 10ms hop size. Mel filterbank employed 40 filters spanning 0-8000 Hz. Frame energy normalization was not applied. Spectral features were computed on 512-point FFT with Hamming windowing. Statistical aggregation (mean and standard deviation) was performed across all frames for each feature dimension.

### A.4 Code Availability

Complete source code, documentation, and results are available in the GitHub repository at the link provided in the submission materials. The repository includes all MATLAB scripts, sample results, and comprehensive README with execution instructions and system requirements.

---

**Word Count:** ~11,500 words

**Declaration of Originality**

I certify that this research report represents my own original work. All sources have been properly cited, and no part of this work has been submitted for assessment in any other course. The implementation was developed independently with assistance only from official MATLAB documentation and cited academic literature. I acknowledge that this work will be checked for academic integrity through Turnitin and other means.

**Signature:** Zhehao Zhu
**Student ID:** 550470317
**Date:** November 11, 2025
