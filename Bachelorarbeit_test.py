{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN+0l7llGB9YR17jkYVLIFP"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e_lMD1uNm-T7",
        "outputId": "800e3487-83d9-42f3-b907-309e31b152f3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (2.15.0)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.4.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=23.5.26 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (23.5.26)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.5.4)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.9.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (16.0.6)\n",
            "Requirement already satisfied: ml-dtypes~=0.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.23.5)\n",
            "Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.3.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow) (23.2)\n",
            "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.20.3)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow) (67.7.2)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.4.0)\n",
            "Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (4.5.0)\n",
            "Requirement already satisfied: wrapt<1.15,>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.14.1)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.35.0)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.60.0)\n",
            "Requirement already satisfied: tensorboard<2.16,>=2.15 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.15.1)\n",
            "Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.15.0)\n",
            "Requirement already satisfied: keras<2.16,>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.15.0)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow) (0.42.0)\n",
            "Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (2.17.3)\n",
            "Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (1.2.0)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (3.5.2)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (2.31.0)\n",
            "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (0.7.2)\n",
            "Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (3.0.1)\n",
            "Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (5.3.2)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (0.3.0)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (4.9)\n",
            "Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow) (1.3.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (2023.11.17)\n",
            "Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow) (2.1.4)\n",
            "Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (0.5.1)\n",
            "Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow) (3.2.2)\n"
          ]
        }
      ],
      "source": [
        "pip install tensorflow"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from google.colab import files\n",
        "import tensorflow as tf\n",
        "from keras.models import Sequential\n",
        "from keras.layers import GRU, Dense, Activation, Masking\n",
        "from sklearn import datasets"
      ],
      "metadata": {
        "id": "3wNf6U7FnGJ5"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Extrahieren die Datensätze für die Nutzung. Vorher müssen die Daten in den Ordner Content eingefügt werden.\n",
        "pfad = \"/content/Blutprobe0.xlsx\"\n",
        "Blutprobe0 = pd.read_excel(pfad)\n",
        "\n",
        "pfad = \"/content/Blutprobe01.xlsx\"\n",
        "Blutprobe01 = pd.read_excel(pfad)\n",
        "\n",
        "pfad = \"/content/Blutprobe02.xlsx\"\n",
        "Blutprobe02 = pd.read_excel(pfad)\n",
        "\n",
        "pfad = \"/content/Blutprobe05.xlsx\"\n",
        "Blutprobe05 = pd.read_excel(pfad)\n",
        "\n",
        "pfad = \"/content/Blutprobe5.xlsx\"\n",
        "Blutprobe5 = pd.read_excel(pfad)"
      ],
      "metadata": {
        "id": "CZC9TqYj2AdO"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Das sind die Ergebnisse die benötigt werden zu jedem Datensatz.\n",
        "ergebnisse = {\n",
        "    'Krankheitszustand': [1, 0, 0, 0, 0]\n",
        "}\n",
        "\n",
        "labels = pd.DataFrame(ergebnisse)"
      ],
      "metadata": {
        "id": "zGULSZOW2fX3",
        "collapsed": true
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Organisieren der Daten und Labels. Verknüpfen die Datensätze jeweils mit einem Label.\n",
        "data_frames = [Blutprobe0, Blutprobe01, Blutprobe02, Blutprobe05, Blutprobe5]\n",
        "data_with_labels = []\n",
        "\n",
        "for i, df in enumerate(data_frames):\n",
        "    label = labels.loc[i, 'Krankheitszustand']  # Das Label für den aktuellen Datensatz\n",
        "    data_with_labels.append((df, label))"
      ],
      "metadata": {
        "id": "s8sT65VI2iNx"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Ausgabe der ersten paar Einträge in data_with_labels\n",
        "for i in range(5):  # Zeige nur die ersten 5 Einträge zur Demonstration\n",
        "    print(\"Datensatz {}: Label - {}\".format(i+1, data_with_labels[i][1]))\n",
        "    print(data_with_labels[i][0])\n",
        "    print()"
      ],
      "metadata": {
        "collapsed": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Xsqe-1082kBo",
        "outputId": "4756e3a2-1e6b-404c-8468-9144ce88187b"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Datensatz 1: Label - 1\n",
            "       Frequenz  Amplitude       Phase  Unbekannt  Unbekannt.1  Unbekannt.2  \\\n",
            "0         12.50   0.033805   37.189231   0.159952   -74.135047     0.211345   \n",
            "1         12.52   0.038209  -79.949534   0.191839  -243.846252     0.199172   \n",
            "2         12.54   0.042465 -103.084784   0.199872    74.089905     0.212459   \n",
            "3         12.56   0.040868 -208.119583   0.183590   -60.300880     0.222604   \n",
            "4         12.58   0.038928 -172.454253   0.175801   -54.756824     0.221434   \n",
            "...         ...        ...         ...        ...          ...          ...   \n",
            "12035      0.00   0.000000    0.000000   0.000000     0.000000     0.000000   \n",
            "12036      0.00   0.000000    0.000000   0.000000     0.000000     0.000000   \n",
            "12037      0.00   0.000000    0.000000   0.000000     0.000000     0.000000   \n",
            "12038      0.00   0.000000    0.000000   0.000000     0.000000     0.000000   \n",
            "12039      0.00   0.000000    0.000000   0.000000     0.000000     0.000000   \n",
            "\n",
            "       Unbekannt.3  Unbekannt.4  \n",
            "0       248.675722  -111.324278  \n",
            "1       196.103283  -163.896717  \n",
            "2       177.174689   177.174689  \n",
            "3       147.818703   147.818703  \n",
            "4       117.697430   117.697430  \n",
            "...            ...          ...  \n",
            "12035     0.000000     0.000000  \n",
            "12036     0.000000     0.000000  \n",
            "12037     0.000000     0.000000  \n",
            "12038     0.000000     0.000000  \n",
            "12039     0.000000     0.000000  \n",
            "\n",
            "[12040 rows x 8 columns]\n",
            "\n",
            "Datensatz 2: Label - 0\n",
            "       Frequenz  Amplitude       Phase  Unbekannt  Unbekannt.1  Unbekannt.2  \\\n",
            "0         12.50   0.029035  141.180152   0.148440    18.825602     0.195603   \n",
            "1         12.52   0.044531  137.957434   0.179882   -12.160373     0.247555   \n",
            "2         12.54   0.041401   10.773507   0.184197  -179.066030     0.224766   \n",
            "3         12.56   0.038474 -249.169788   0.170577   252.328333     0.225552   \n",
            "4         12.58   0.037815 -161.741492   0.168347   -51.432966     0.224624   \n",
            "...         ...        ...         ...        ...          ...          ...   \n",
            "12035     18.42   0.094208  172.356285   0.747224    49.356472     0.126078   \n",
            "12036     18.44   0.100858 -174.806812   0.757083    27.215652     0.133220   \n",
            "12037     18.46   0.100603  229.907425   0.788996    38.348908     0.127507   \n",
            "12038     18.48   0.090762  243.766493   0.808910    41.375588     0.112202   \n",
            "12039     18.50   0.133678  -98.726848   0.780204    31.605696     0.171337   \n",
            "\n",
            "       Unbekannt.3  Unbekannt.4  \n",
            "0       237.645451  -122.354550  \n",
            "1       209.882192  -150.117808  \n",
            "2       170.160463  -189.839537  \n",
            "3       141.498121   501.498121  \n",
            "4       110.308526   110.308526  \n",
            "...            ...          ...  \n",
            "12035   237.000187  -122.999813  \n",
            "12036   202.022464   202.022464  \n",
            "12037   168.441483  -191.558517  \n",
            "12038   157.609094  -202.390906  \n",
            "12039   130.332544   130.332544  \n",
            "\n",
            "[12040 rows x 8 columns]\n",
            "\n",
            "Datensatz 3: Label - 0\n",
            "       Frequenz  Amplitude       Phase  Unbekannt  Unbekannt.1  Unbekannt.2  \\\n",
            "0         12.50   0.037484  -51.921743   0.151270  -172.139924     0.247798   \n",
            "1         12.52   0.041118  -74.687785   0.182489   133.165826     0.225320   \n",
            "2         12.54   0.044560   10.101773   0.192166  -179.714171     0.231883   \n",
            "3         12.56   0.040855  -47.970356   0.175621   100.488105     0.232633   \n",
            "4         12.58   0.038283 -226.338730   0.170837  -110.052849     0.224089   \n",
            "...         ...        ...         ...        ...          ...          ...   \n",
            "12035     18.42   0.099713 -222.097166   0.756925     3.889300     0.131734   \n",
            "12036     18.44   0.108526  228.237075   0.777631    60.708166     0.139560   \n",
            "12037     18.46   0.108052 -142.515923   0.804898    17.874593     0.134243   \n",
            "12038     18.48   0.093282 -113.956624   0.825389    34.565576     0.113016   \n",
            "12039     18.50   0.135993  269.111095   0.795283    34.634896     0.170999   \n",
            "\n",
            "       Unbekannt.3  Unbekannt.4  \n",
            "0       239.781820  -120.218181  \n",
            "1       207.853611   207.853611  \n",
            "2       170.184056  -189.815944  \n",
            "3       148.458461   148.458461  \n",
            "4       116.285881   116.285881  \n",
            "...            ...          ...  \n",
            "12035   225.986465   225.986465  \n",
            "12036   192.471091  -167.528909  \n",
            "12037   160.390517   160.390517  \n",
            "12038   148.522200   148.522200  \n",
            "12039   125.523801  -234.476199  \n",
            "\n",
            "[12040 rows x 8 columns]\n",
            "\n",
            "Datensatz 4: Label - 0\n",
            "       Frequenz  Amplitude       Phase  Unbekannt  Unbekannt.1  Unbekannt.2  \\\n",
            "0         12.50   0.037167   16.330861   0.147743   -76.583214     0.251567   \n",
            "1         12.52   0.039084  181.187991   0.182660    42.412723     0.213970   \n",
            "2         12.54   0.043642   22.296195   0.189034  -156.934140     0.230869   \n",
            "3         12.56   0.041410  108.190740   0.171114   258.931998     0.242005   \n",
            "4         12.58   0.039490   27.506050   0.168488  -212.293808     0.234382   \n",
            "...         ...        ...         ...        ...          ...          ...   \n",
            "12035     18.42   0.104614  137.114628   0.741490    14.581256     0.141086   \n",
            "12036     18.44   0.111805 -196.069116   0.760746     6.381078     0.146968   \n",
            "12037     18.46   0.110611  235.317960   0.790539    45.499471     0.139919   \n",
            "12038     18.48   0.101908 -118.334897   0.813579    40.740628     0.125259   \n",
            "12039     18.50   0.143100  -90.304850   0.782050    42.145719     0.182980   \n",
            "\n",
            "       Unbekannt.3  Unbekannt.4  \n",
            "0       267.085925   -92.914075  \n",
            "1       221.224732  -138.775268  \n",
            "2       180.769664  -179.230336  \n",
            "3       150.741258   150.741258  \n",
            "4       120.200142  -239.799858  \n",
            "...            ...          ...  \n",
            "12035   237.466627  -122.533372  \n",
            "12036   202.450194   202.450194  \n",
            "12037   170.181511  -189.818489  \n",
            "12038   159.075525   159.075525  \n",
            "12039   132.450570   132.450570  \n",
            "\n",
            "[12040 rows x 8 columns]\n",
            "\n",
            "Datensatz 5: Label - 0\n",
            "       Frequenz  Amplitude       Phase  Unbekannt  Unbekannt.1  Unbekannt.2  \\\n",
            "0         12.50   0.036604  -51.994941   0.144761   189.766543     0.252861   \n",
            "1         12.52   0.045448  -15.275167   0.183750   184.446136     0.247334   \n",
            "2         12.54   0.047609 -178.555937   0.186423    -7.471443     0.255380   \n",
            "3         12.56   0.042418   10.676835   0.170906   157.056651     0.248193   \n",
            "4         12.58   0.040777  -96.620344   0.167994    20.526837     0.242727   \n",
            "...         ...        ...         ...        ...          ...          ...   \n",
            "12035     18.42   0.094446 -206.189210   0.741477    34.821979     0.127375   \n",
            "12036     18.44   0.100630 -175.398013   0.755602    29.666990     0.133179   \n",
            "12037     18.46   0.099705  232.930633   0.784754    45.721716     0.127053   \n",
            "12038     18.48   0.092696 -120.253565   0.809365    41.787577     0.114529   \n",
            "12039     18.50   0.136335  261.149253   0.779626    36.679325     0.174873   \n",
            "\n",
            "       Unbekannt.3  Unbekannt.4  \n",
            "0       241.761484   241.761484  \n",
            "1       199.721303   199.721303  \n",
            "2       171.084494   171.084494  \n",
            "3       146.379816   146.379816  \n",
            "4       117.147181   117.147181  \n",
            "...            ...          ...  \n",
            "12035   241.011188   241.011188  \n",
            "12036   205.065003   205.065003  \n",
            "12037   172.791084  -187.208916  \n",
            "12038   162.041141   162.041141  \n",
            "12039   135.530073  -224.469927  \n",
            "\n",
            "[12040 rows x 8 columns]\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = np.array([df.values for df, _ in data_with_labels[-4:]])  # Extrahieren der Daten aus den DataFrames\n",
        "X_train = X_train[:, :, [0, 5, 6]]   # Die ersten drei Spalten werden extrahiert (Frequenz, Amplitude, Phase)\n",
        "y_train = np.array([label for _, label in data_with_labels[-4:]])  # Extrahieren der Labels\n",
        "X_test = np.array([Blutprobe0.values])\n",
        "X_test = X_test[:, :, [0, 5, 6]]\n",
        "label_entry = labels['Krankheitszustand'].iloc[0]\n",
        "y_test = np.array(label_entry)"
      ],
      "metadata": {
        "id": "DbPBe2i02nFQ"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Netzwerksparameter\n",
        "input_size = 3\n",
        "hidden_size = 256\n",
        "num_classes = 1\n",
        "batch_size = 14\n",
        "num_epochs = 50\n",
        "learning_rate = 0.0003\n",
        "threshold = 0.5"
      ],
      "metadata": {
        "id": "i6-eEmsj6-cW"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Modell erstellen\n",
        "model = Sequential([\n",
        "    Masking(mask_value=0, input_shape=(12040, input_size)),\n",
        "    GRU(hidden_size, input_shape=(12040, input_size), return_sequences=True),\n",
        "    GRU(hidden_size, return_sequences=True),\n",
        "    GRU(hidden_size, return_sequences=False),\n",
        "    Dense(hidden_size, input_shape=(hidden_size,), activation='relu'),\n",
        "    Dense(hidden_size, input_shape=(hidden_size,), activation='relu'),\n",
        "    Dense(num_classes, activation='sigmoid')\n",
        "])"
      ],
      "metadata": {
        "collapsed": true,
        "id": "KajiXY2I7AWL",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "274578ed-951c-4942-e7fe-9a1269d9a085"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/keras/src/layers/core/masking.py:47: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(**kwargs)\n",
            "/usr/local/lib/python3.10/dist-packages/keras/src/layers/rnn/rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(**kwargs)\n",
            "/usr/local/lib/python3.10/dist-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Loss-Funktion, Optimizer\n",
        "optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)\n",
        "model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "TQYtA7z07DCH"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "history = model.fit(X_train, y_train,epochs= num_epochs, batch_size= batch_size)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3qbnqIKJ7oJU",
        "outputId": "6ab291d1-6cb7-4dcc-95cb-925b214d92fa"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m97s\u001b[0m 97s/step - accuracy: 1.0000 - loss: 0.2386\n",
            "Epoch 2/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m141s\u001b[0m 141s/step - accuracy: 1.0000 - loss: 0.0970\n",
            "Epoch 3/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m139s\u001b[0m 139s/step - accuracy: 1.0000 - loss: 0.0369\n",
            "Epoch 4/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m142s\u001b[0m 142s/step - accuracy: 1.0000 - loss: 0.0126\n",
            "Epoch 5/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m143s\u001b[0m 143s/step - accuracy: 1.0000 - loss: 0.0043\n",
            "Epoch 6/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m91s\u001b[0m 91s/step - accuracy: 1.0000 - loss: 0.0016\n",
            "Epoch 7/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m138s\u001b[0m 138s/step - accuracy: 1.0000 - loss: 6.1453e-04\n",
            "Epoch 8/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m165s\u001b[0m 165s/step - accuracy: 1.0000 - loss: 2.5896e-04\n",
            "Epoch 9/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m119s\u001b[0m 119s/step - accuracy: 1.0000 - loss: 1.1695e-04\n",
            "Epoch 10/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m142s\u001b[0m 142s/step - accuracy: 1.0000 - loss: 5.6264e-05\n",
            "Epoch 11/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m140s\u001b[0m 140s/step - accuracy: 1.0000 - loss: 2.9039e-05\n",
            "Epoch 12/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m142s\u001b[0m 142s/step - accuracy: 1.0000 - loss: 1.5963e-05\n",
            "Epoch 13/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m141s\u001b[0m 141s/step - accuracy: 1.0000 - loss: 9.2742e-06\n",
            "Epoch 14/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m144s\u001b[0m 144s/step - accuracy: 1.0000 - loss: 5.6605e-06\n",
            "Epoch 15/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m143s\u001b[0m 143s/step - accuracy: 1.0000 - loss: 3.6181e-06\n",
            "Epoch 16/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m162s\u001b[0m 162s/step - accuracy: 1.0000 - loss: 2.4090e-06\n",
            "Epoch 17/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m138s\u001b[0m 138s/step - accuracy: 1.0000 - loss: 1.6633e-06\n",
            "Epoch 18/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m126s\u001b[0m 126s/step - accuracy: 1.0000 - loss: 1.1868e-06\n",
            "Epoch 19/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m85s\u001b[0m 85s/step - accuracy: 1.0000 - loss: 8.7301e-07\n",
            "Epoch 20/20\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m144s\u001b[0m 144s/step - accuracy: 1.0000 - loss: 6.6099e-07\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Plotten die Veränderung der Loss abhängig von Epoch.\n",
        "plt.plot(history.history['loss'], label='Trainingsverlust')\n",
        "plt.xlabel('Epochen')\n",
        "plt.ylabel('Loss')\n",
        "plt.title('Verlust während des Trainings')\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "VR0Cl2S5Yj_e",
        "outputId": "da74e68c-8e5e-4323-ac79-5b6e08c174d7"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(history.history['accuracy'], label='Trainingsgenauigkeit')\n",
        "#plt.plot(history.history['val_accuracy'], label='Validierungsgenauigkeit')\n",
        "plt.xlabel('Epochen')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.title('Genauigkeit während des Trainings')\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "HDA4Evi6Ykol",
        "outputId": "1504ec2b-7716-448d-a2c6-48810a957449"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkAAAAHHCAYAAABXx+fLAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABHuUlEQVR4nO3deXyNZ/7/8fdJyMlCEmsiloilltJYioZam4qlBtVaxlTQjjKWqjGWqb3VtFqtFlXd0A2tYnyrZUilyijVoFQZNJYqiaUSCSKS6/dHfznTI4scEkncr+fjcR4P57qv+z6f677Pcd65t2MzxhgBAABYiFthFwAAAHC7EYAAAIDlEIAAAIDlEIAAAIDlEIAAAIDlEIAAAIDlEIAAAIDlEIAAAIDlEIAAAIDlEICAIuLo0aOy2WxavHjxTc/78ssv50stAwcOVPXq1fNlWa5q0KCB2rVr56ijVKlSWfrk1F5UVa9eXQMHDrypeW/lfVGUTZs2TTab7abmXbx4sWw2m44ePZq/RcFSCEAotuLi4jRixAjddddd8vb2lre3t+rXr6/hw4frhx9+KOzy7iiXLl3StGnTFBMTU+Cv9cILL+iZZ56RJD355JN69913C/w18T/Vq1eXzWa74eNOC2SwnhKFXQBwMz7//HP16dNHJUqUUP/+/RUaGio3NzcdOHBAK1eu1IIFCxQXF6fg4ODCLjXPgoODdfnyZZUsWbKwS9Hbb7+tjIwMx/NLly5p+vTpkuTYO1NQHnroIce/w8LCFBYWVqCvB2dz5sxRcnKy4/kXX3yhpUuX6tVXX1X58uUd7S1btryl15k0aZImTJhwU/M+9thj6tu3r+x2+y3VAGsjAKHYOXLkiPr27avg4GBFR0erUqVKTtNffPFFvfHGG3JzK147OG02mzw9PQu7DEkqEiGsoFy7dk0ZGRny8PAo7FKKpB49ejg9P336tJYuXaoePXrkelg0JSVFPj4+eX6dEiVKqESJm/sKcnd3l7u7+03NC2QqXt8QgKRZs2YpJSVFixYtyhJ+pN//Yx01apSqVq3q1H7gwAE98sgjKlu2rDw9PXXvvfdqzZo1Tn0yzy3YunWrxowZowoVKsjHx0c9e/bUmTNnnPr+61//UteuXRUUFCS73a6aNWvq2WefVXp6ulO/nM7/aNeundPelJzO9fj0009Vv359eXp6qkGDBlq1alWeztExxmjIkCHy8PDQypUrHe0ffvihmjZtKi8vL5UtW1Z9+/bViRMnnOb94/KPHj2qChUqSJKmT5/uOAQybdq0bF/3woULcnd31+uvv+5oO3v2rNzc3FSuXDkZYxztw4YNU2BgoOP5N998o0cffVTVqlWT3W5X1apV9fTTT+vy5cvZvtbJkyfVo0cPlSpVShUqVNDYsWOd1v8fz42aM2eOatasKbvdrv3790vK//eEMUbPPfecqlSpIm9vb7Vv314//vhjtrXntO4GDhwoPz8/+fv7KzIyUhcuXMi2b15qT0tL0/Tp01W7dm15enqqXLlyuv/++7Vhw4Y815SdzHOwjhw5oi5duqh06dLq37+/pLxvw+zOAbLZbBoxYoRWr16tBg0ayG636+6779a6deuc+mV3DlD16tX10EMPacuWLWrevLk8PT1Vo0YNvf/++1nq/+GHH9S2bVt5eXmpSpUqeu6557Ro0aIsy9y5c6ciIiJUvnx5eXl5KSQkRIMHD76ldYeigz1AKHY+//xz1apVSy1atMjzPD/++KNatWqlypUra8KECfLx8dEnn3yiHj166LPPPlPPnj2d+o8cOVJlypTR1KlTdfToUc2ZM0cjRozQ8uXLHX0WL16sUqVKacyYMSpVqpS++uorTZkyRUlJSXrppZfyZaxr165Vnz591LBhQ0VFRem3337T448/rsqVK+c6X3p6ugYPHqzly5dr1apV6tq1qyRp5syZmjx5snr37q0nnnhCZ86c0dy5c9WmTRvt2rVL/v7+WZZVoUIFLViwQMOGDVPPnj318MMPS5LuueeebF/b399fDRo00ObNmzVq1ChJ0pYtW2Sz2XT+/Hnt379fd999t6Tfvyxbt27tmPfTTz9VSkqKhg0bpnLlymnHjh2aO3eufvnlF3366adZxhgREaEWLVro5Zdf1saNGzV79mzVrFlTw4YNc+q7aNEiXblyRUOGDJHdblfZsmUL5D0xZcoUPffcc+rSpYu6dOmi2NhYdezYUVevXs11e0m/h6fu3btry5YtGjp0qOrVq6dVq1YpMjIyS9+81j5t2jRFRUXpiSeeUPPmzZWUlKSdO3cqNjZWDz744A1rys21a9cUERGh+++/Xy+//LK8vb0l/b4NL126lKdtmJ0tW7Zo5cqV+tvf/qbSpUvr9ddfV69evXT8+HGVK1cu13kPHz6sRx55RI8//rgiIyP13nvvaeDAgWratKnjPXfy5Em1b99eNptNEydOlI+Pj955550sh9MSEhLUsWNHVahQQRMmTJC/v7+OHj3q9McEijkDFCOJiYlGkunRo0eWab/99ps5c+aM43Hp0iXHtAceeMA0bNjQXLlyxdGWkZFhWrZsaWrXru1oW7RokZFkwsPDTUZGhqP96aefNu7u7ubChQuOtj8uP9OTTz5pvL29nV4nODjYREZGZunbtm1b07ZtW8fzuLg4I8ksWrTI0dawYUNTpUoVc/HiRUdbTEyMkWSCg4OzzPvSSy+ZtLQ006dPH+Pl5WXWr1/v6HP06FHj7u5uZs6c6VTH3r17TYkSJZzaIyMjnZZ/5swZI8lMnTo1yziyM3z4cBMQEOB4PmbMGNOmTRtTsWJFs2DBAmOMMefOnTM2m8289tprjn4pKSlZlhUVFWVsNps5duyYU32SzIwZM5z6Nm7c2DRt2jTLevH19TUJCQlOffP7PZGQkGA8PDxM165dnfr985//NJKyfQ/80erVq40kM2vWLEfbtWvXTOvWrbO8L/Jae2hoqOnatWuur3sjL730kpFk4uLiHG2Z63/ChAlZ+mf3uchuG06dOtVc/xUkyXh4eJjDhw872vbs2WMkmblz5zraMrfJH2sKDg42kszmzZsdbQkJCcZut5u///3vjraRI0cam81mdu3a5Wg7d+6cKVu2rNMyV61aZSSZ7777LueVg2KNQ2AoVpKSkiQp20ug27VrpwoVKjge8+fPlySdP39eX331lXr37q2LFy/q7NmzOnv2rM6dO6eIiAgdOnRIJ0+edFrWkCFDnHbPt27dWunp6Tp27JijzcvLy/HvzOW2bt1aly5d0oEDB255rL/++qv27t2rAQMGOI23bdu2atiwYbbzXL16VY8++qg+//xzffHFF+rYsaNj2sqVK5WRkaHevXs71sHZs2cVGBio2rVra9OmTbdcc6bWrVsrPj5eBw8elPT7np42bdqodevW+uabbyT9/pe+McZpD1DmXgTp93NKzp49q5YtW8oYo127dmV5naFDh2Z53Z9//jlLv169ejkO40kF857YuHGjrl69qpEjRzr1Gz169A3Xl/T7ycYlSpRw2nvl7u6ukSNHOvVzpXZ/f3/9+OOPOnToUJ5qcNX1e9ok589FXrbh9cLDw1WzZk3H83vuuUe+vr7Zbtfr1a9f3+n9VKFCBdWpU8dp3nXr1iksLEyNGjVytJUtW9ZxCC9T5t7Qzz//XGlpaTd8bRQ/HAJDsVK6dGlJcrpKJdPChQt18eJFxcfH6y9/+Yuj/fDhwzLGaPLkyZo8eXK2y01ISHA6rFStWjWn6WXKlJEk/fbbb462H3/8UZMmTdJXX33lCGaZEhMTXRxZVplfrLVq1coyrVatWoqNjc3SHhUVpeTkZH355ZdZrtY6dOiQjDGqXbt2tq+Xnyc+Z34JffPNN6pSpYp27dql5557ThUqVHDcq+ibb76Rr6+vQkNDHfMdP35cU6ZM0Zo1a5zWtZR1nXp6ejqFGun37XT9fJIUEhLi9Lwg3hOZ2+v69VuhQgVH39wcO3ZMlSpVyhLu69Spc9O1z5gxQ927d9ddd92lBg0aqFOnTnrsscdyPHzpihIlSqhKlSpZ2l3Zhtm5fj1LOW/Xm5n32LFj2V5ZeP3nrG3bturVq5emT5+uV199Ve3atVOPHj305z//mavP7hAEIBQrfn5+qlSpkvbt25dlWuY5QdffHC3zcu6xY8cqIiIi2+Ve/59fTleYmP9/Au+FCxfUtm1b+fr6asaMGapZs6Y8PT0VGxur8ePHO11CntPN3tLT0/P9SpaIiAitW7dOs2bNUrt27ZyuKsvIyJDNZtOXX36Z7evm540Fg4KCFBISos2bN6t69eoyxigsLEwVKlTQU089pWPHjumbb75Ry5YtHVfrpaen68EHH9T58+c1fvx41a1bVz4+Pjp58qQGDhzotE6lnLdRdv64V0IqmPfE7eJK7W3atNGRI0f0r3/9S//+97/1zjvv6NVXX9Wbb76pJ5544pbqsNvtWa60dHUbZudW1nN+biObzaYVK1bo22+/1f/93/9p/fr1Gjx4sGbPnq1vv/22WN2IE9kjAKHY6dq1q9555x3t2LFDzZs3v2H/GjVqSPp9D0d4eHi+1BATE6Nz585p5cqVatOmjaM9Li4uS98yZcpkeyXPsWPHHLVlJ/MeRocPH84yLbs2Sbrvvvs0dOhQPfTQQ3r00Ue1atUqx6XGNWvWlDFGISEhuuuuu3Id3/Vu5o69rVu31ubNmxUSEqJGjRqpdOnSCg0NlZ+fn9atW6fY2FjHvYUkae/evfrvf/+rJUuWaMCAAY72W71iKTsF8Z7I3F6HDh1y2q5nzpzJ096LzNs6JCcnO325Zh5GzORq7WXLltWgQYM0aNAgJScnq02bNpo2bdotB6Ds3M5teLOCg4Nd/kzdd999mjlzpj7++GP1799fy5YtK5D1h9uLc4BQ7IwbN07e3t4aPHiw4uPjs0y//q+9ihUrql27dlq4cKFOnTqVpf/1lzLnReZfmn98ratXr+qNN97I0rdmzZr69ttvna4E+vzzz7Ncen69oKAgNWjQQO+//77TIb+vv/5ae/fuzXG+8PBwLVu2TOvWrdNjjz3m+Kv74Ycflru7u6ZPn55lHRljdO7cuRyXmXluTk6XZGendevWOnr0qJYvX+44JObm5qaWLVvqlVdeUVpamtP5GtmtU2OMXnvttTy/Zl4VxHsiPDxcJUuW1Ny5c53GMGfOnDzN36VLF127dk0LFixwtKWnp2vu3Lk3Xfv127RUqVKqVauWUlNT81STq27nNrxZERER2rZtm3bv3u1oO3/+vD766COnfr/99luWz0nmeUMFtf5we7EHCMVO7dq19fHHH6tfv36qU6eO407QxhjFxcXp448/lpubm9P5CfPnz9f999+vhg0b6q9//atq1Kih+Ph4bdu2Tb/88ov27NnjUg0tW7ZUmTJlFBkZqVGjRslms+mDDz7Idlf7E088oRUrVqhTp07q3bu3jhw5og8//NDpRM+cPP/88+revbtatWqlQYMG6bffftO8efPUoEGDbM+DytSjRw8tWrRIAwYMkK+vrxYuXKiaNWvqueee08SJE3X06FH16NFDpUuXVlxcnFatWqUhQ4Zo7Nix2S7Py8tL9evX1/Lly3XXXXepbNmyatCggRo0aJBjDZnh5uDBg3r++ecd7W3atNGXX34pu92uZs2aOdrr1q2rmjVrauzYsTp58qR8fX312Wef5Wnvyc3I7/dE5n2IoqKi9NBDD6lLly7atWuXvvzyS6c7KOekW7duatWqlSZMmKCjR4+qfv36WrlyZbbnzeS19vr166tdu3Zq2rSpypYtq507d2rFihUaMWKES2PLq9u9DW/GuHHj9OGHH+rBBx/UyJEjHZfBV6tWTefPn3fs7VyyZIneeOMN9ezZUzVr1tTFixf19ttvy9fXV126dCnkUSBf3LbrzYB8dvjwYTNs2DBTq1Yt4+npaby8vEzdunXN0KFDze7du7P0P3LkiBkwYIAJDAw0JUuWNJUr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io2N1XfffZfn5aampio1NdXxPCkpKd9qBgAARU+x2gN0IydOnNBTTz2ljz76SJ6ennmeLyoqSn5+fo5H1apVC7BKAABQ2IpVAAoMDFR8fLxTW3x8vHx9feXl5aXvv/9eCQkJatKkiUqUKKESJUro66+/1uuvv64SJUooPT092+VOnDhRiYmJjseJEydux3AAAEAhKVaHwMLCwvTFF184tW3YsEFhYWGSpAceeEB79+51mj5o0CDVrVtX48ePl7u7e7bLtdvtstvtBVM0AAAocgo1ACUnJ+vw4cOO53Fxcdq9e7fKli2ratWqaeLEiTp58qTef/99SdLQoUM1b948jRs3ToMHD9ZXX32lTz75RGvXrpUklS5dWg0aNHB6DR8fH5UrVy5LOwAAsK5CPQS2c+dONW7cWI0bN5YkjRkzRo0bN9aUKVMkSadOndLx48cd/UNCQrR27Vpt2LBBoaGhmj17tt555x2nS+ABAABuxGaMMYVdRFGTlJQkPz8/JSYmytfXt7DLAQAAeeDK93exOgkaAAAgPxCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RCAAACA5RRqANq8ebO6deumoKAg2Ww2rV69+obzxMTEqEmTJrLb7apVq5YWL17sND0qKkrNmjVT6dKlVbFiRfXo0UMHDx4smAEAAIBiqVADUEpKikJDQzV//vw89Y+Li1PXrl3Vvn177d69W6NHj9YTTzyh9evXO/p8/fXXGj58uL799ltt2LBBaWlp6tixo1JSUgpqGAAAoJixGWNMYRchSTabTatWrVKPHj1y7DN+/HitXbtW+/btc7T17dtXFy5c0Lp167Kd58yZM6pYsaK+/vprtWnTJk+1JCUlyc/PT4mJifL19XVpHAAAoHC48v1drM4B2rZtm8LDw53aIiIitG3bthznSUxMlCSVLVu2QGsDAADFR4nCLsAVp0+fVkBAgFNbQECAkpKSdPnyZXl5eTlNy8jI0OjRo9WqVSs1aNAgx+WmpqYqNTXV8TwpKSl/CwcAAEVKsdoD5Krhw4dr3759WrZsWa79oqKi5Ofn53hUrVr1NlUIAAAKQ7EKQIGBgYqPj3dqi4+Pl6+vb5a9PyNGjNDnn3+uTZs2qUqVKrkud+LEiUpMTHQ8Tpw4ke+1AwCAoqNYHQILCwvTF1984dS2YcMGhYWFOZ4bYzRy5EitWrVKMTExCgkJueFy7Xa77HZ7vtcLAACKpkLdA5ScnKzdu3dr9+7dkn6/zH337t06fvy4pN/3zAwYMMDRf+jQofr55581btw4HThwQG+88YY++eQTPf30044+w4cP14cffqiPP/5YpUuX1unTp3X69Gldvnz5to4NAAAUXYV6GXxMTIzat2+fpT0yMlKLFy/WwIEDdfToUcXExDjN8/TTT2v//v2qUqWKJk+erIEDBzqm22y2bF9r0aJFTv1yw2XwAAAUP658fxeZ+wAVJQQgAACKnzv2PkAAAAD5gQAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAshwAEAAAsx+UAVL16dc2YMUPHjx8viHoAAAAKnMsBaPTo0Vq5cqVq1KihBx98UMuWLVNqampB1AYAAFAgbioA7d69Wzt27FC9evU0cuRIVapUSSNGjFBsbGxB1AgAAJCvbMYYcysLSEtL0xtvvKHx48crLS1NDRs21KhRozRo0CDZbLb8qvO2SkpKkp+fnxITE+Xr61vY5QAAgDxw5fu7xM2+SFpamlatWqVFixZpw4YNuu+++/T444/rl19+0T//+U9t3LhRH3/88c0uHgAAoMC4HIBiY2O1aNEiLV26VG5ubhowYIBeffVV1a1b19GnZ8+eatasWb4WCgAAkF9cDkDNmjXTgw8+qAULFqhHjx4qWbJklj4hISHq27dvvhQIAACQ31wOQD///LOCg4Nz7ePj46NFixbddFEAAAAFyeWrwBISErR9+/Ys7du3b9fOnTvzpSgAAICC5HIAGj58uE6cOJGl/eTJkxo+fHi+FAUAAFCQXA5A+/fvV5MmTbK0N27cWPv378+XogAAAAqSywHIbrcrPj4+S/upU6dUosRNX1UPAABw27gcgDp27KiJEycqMTHR0XbhwgX985//1IMPPpivxQEAABQEl3fZvPzyy2rTpo2Cg4PVuHFjSdLu3bsVEBCgDz74IN8LBAAAyG8uB6DKlSvrhx9+0EcffaQ9e/bIy8tLgwYNUr9+/bK9JxAAAEBRc1Mn7fj4+GjIkCH5XQsAAMBtcdNnLe/fv1/Hjx/X1atXndr/9Kc/3XJRAAAABemm7gTds2dP7d27VzabTZk/Jp/5y+/p6en5WyEAAEA+c/kqsKeeekohISFKSEiQt7e3fvzxR23evFn33nuvYmJiCqBEAACA/OXyHqBt27bpq6++Uvny5eXm5iY3Nzfdf//9ioqK0qhRo7Rr166CqBMAACDfuLwHKD09XaVLl5YklS9fXr/++qskKTg4WAcPHszf6gAAAAqAy3uAGjRooD179igkJEQtWrTQrFmz5OHhobfeeks1atQoiBoBAADylcsBaNKkSUpJSZEkzZgxQw899JBat26tcuXKafny5fleIAAAQH6zmczLuG7B+fPnVaZMGceVYMVdUlKS/Pz8lJiYKF9f38IuBwAA5IEr398unQOUlpamEiVKaN++fU7tZcuWvWPCDwAAuPO5FIBKliypatWq5du9fjZv3qxu3bopKChINptNq1evvuE8MTExatKkiex2u2rVqqXFixdn6TN//nxVr15dnp6eatGihXbs2JEv9QIAgDuDy1eBPfPMM/rnP/+p8+fP3/KLp6SkKDQ0VPPnz89T/7i4OHXt2lXt27fX7t27NXr0aD3xxBNav369o8/y5cs1ZswYTZ06VbGxsQoNDVVERIQSEhJuuV4AAHBncPkcoMaNG+vw4cNKS0tTcHCwfHx8nKbHxsbeXCE2m1atWqUePXrk2Gf8+PFau3at0yG4vn376sKFC1q3bp0kqUWLFmrWrJnmzZsnScrIyFDVqlU1cuRITZgwIU+1FNQ5QMYYXU7jTtkAAEiSV0n3fD2FxpXvb5evAsstoBS0bdu2KTw83KktIiJCo0ePliRdvXpV33//vSZOnOiY7ubmpvDwcG3bti3H5aampio1NdXxPCkpKX8L//8up6Wr/pT1N+4IAIAF7J8RIW+Pm/5Z0lvi8qtOnTq1IOrIk9OnTysgIMCpLSAgQElJSbp8+bJ+++03paenZ9vnwIEDOS43KipK06dPL5CaAQBA0VM4sauImThxosaMGeN4npSUpKpVq+b763iVdNf+GRH5vlwAAIojr5LuhfbaLgcgNze3XI/XFeSvwQcGBio+Pt6pLT4+Xr6+vvLy8pK7u7vc3d2z7RMYGJjjcu12u+x2e4HU/Ec2m63QdvUBAID/cfnbeNWqVU7P09LStGvXLi1ZsqTADyOFhYXpiy++cGrbsGGDwsLCJEkeHh5q2rSpoqOjHecqZWRkKDo6WiNGjCjQ2gAAQPHhcgDq3r17lrZHHnlEd999t5YvX67HH388z8tKTk7W4cOHHc/j4uK0e/dulS1bVtWqVdPEiRN18uRJvf/++5KkoUOHat68eRo3bpwGDx6sr776Sp988onWrl3rWMaYMWMUGRmpe++9V82bN9ecOXOUkpKiQYMGuTpUAABwh8q34zH33XefhgwZ4tI8O3fuVPv27R3PM8/DiYyM1OLFi3Xq1CkdP37cMT0kJERr167V008/rddee01VqlTRO++8o4iI/51X06dPH505c0ZTpkzR6dOn1ahRI61bty7LidEAAMC68uW3wC5fvqyJEyfqyy+/1MGDB/OjrkLFb4EBAFD8FOh9gK7/0VNjjC5evChvb299+OGHrlcLAABwm7kcgF599VWnAOTm5qYKFSqoRYsWKlOmTL4WBwAAUBBcDkADBw4sgDIAAABuH5d/DHXRokX69NNPs7R/+umnWrJkSb4UBQAAUJBcDkBRUVEqX758lvaKFSvq+eefz5eiAAAACpLLAej48eMKCQnJ0h4cHOx0yToAAEBR5XIAqlixon744Ycs7Xv27FG5cuXypSgAAICC5HIA6tevn0aNGqVNmzYpPT1d6enp+uqrr/TUU0+pb9++BVEjAABAvnL5KrBnn31WR48e1QMPPKASJX6fPSMjQwMGDOAcIAAAUCzc9J2gDx06pN27d8vLy0sNGzZUcHBwftdWaLgTNAAAxU+B3gk6U+3atVW7du2bnR0AAKDQuHwOUK9evfTiiy9maZ81a5YeffTRfCkKAACgILkcgDZv3qwuXbpkae/cubM2b96cL0UBAAAUJJcDUHJysjw8PLK0lyxZUklJSflSFAAAQEFyOQA1bNhQy5cvz9K+bNky1a9fP1+KAgAAKEgunwQ9efJkPfzwwzpy5Ig6dOggSYqOjtbHH3+sFStW5HuBAAAA+c3lANStWzetXr1azz//vFasWCEvLy+Fhobqq6++UtmyZQuiRgAAgHx10/cBypSUlKSlS5fq3Xff1ffff6/09PT8qq3QcB8gAACKH1e+v10+ByjT5s2bFRkZqaCgIM2ePVsdOnTQt99+e7OLAwAAuG1cOgR2+vRpLV68WO+++66SkpLUu3dvpaamavXq1ZwADQAAio087wHq1q2b6tSpox9++EFz5szRr7/+qrlz5xZkbQAAAAUiz3uAvvzyS40aNUrDhg3jJzAAAECxluc9QFu2bNHFixfVtGlTtWjRQvPmzdPZs2cLsjYAAIACkecAdN999+ntt9/WqVOn9OSTT2rZsmUKCgpSRkaGNmzYoIsXLxZknQAAAPnmli6DP3jwoN5991198MEHunDhgh588EGtWbMmP+srFFwGDwBA8XNbLoOXpDp16mjWrFn65ZdftHTp0ltZFAAAwG1zyzdCvBOxBwgAgOLntu0BAgAAKI4IQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIIQAAAwHIKPQDNnz9f1atXl6enp1q0aKEdO3bk2DctLU0zZsxQzZo15enpqdDQUK1bt86pT3p6uiZPnqyQkBB5eXmpZs2aevbZZ2WMKeihAACAYqJQA9Dy5cs1ZswYTZ06VbGxsQoNDVVERIQSEhKy7T9p0iQtXLhQc+fO1f79+zV06FD17NlTu3btcvR58cUXtWDBAs2bN08//fSTXnzxRc2aNUtz5869XcMCAABFnM0U4q6RFi1aqFmzZpo3b54kKSMjQ1WrVtXIkSM1YcKELP2DgoL0zDPPaPjw4Y62Xr16ycvLSx9++KEk6aGHHlJAQIDefffdHPvcSFJSkvz8/JSYmChfX99bGSIAALhNXPn+LrQ9QFevXtX333+v8PDw/xXj5qbw8HBt27Yt23lSU1Pl6enp1Obl5aUtW7Y4nrds2VLR0dH673//K0nas2ePtmzZos6dO+dYS2pqqpKSkpweAADgzlWisF747NmzSk9PV0BAgFN7QECADhw4kO08EREReuWVV9SmTRvVrFlT0dHRWrlypdLT0x19JkyYoKSkJNWtW1fu7u5KT0/XzJkz1b9//xxriYqK0vTp0/NnYAAAoMgr9JOgXfHaa6+pdu3aqlu3rjw8PDRixAgNGjRIbm7/G8Ynn3yijz76SB9//LFiY2O1ZMkSvfzyy1qyZEmOy504caISExMdjxMnTtyO4QAAgEJSaHuAypcvL3d3d8XHxzu1x8fHKzAwMNt5KlSooNWrV+vKlSs6d+6cgoKCNGHCBNWoUcPR5x//+IcmTJigvn37SpIaNmyoY8eOKSoqSpGRkdku1263y26359PIAABAUVdoe4A8PDzUtGlTRUdHO9oyMjIUHR2tsLCwXOf19PRU5cqVde3aNX322Wfq3r27Y9qlS5ec9ghJkru7uzIyMvJ3AAAAoNgqtD1AkjRmzBhFRkbq3nvvVfPmzTVnzhylpKRo0KBBkqQBAwaocuXKioqKkiRt375dJ0+eVKNGjXTy5ElNmzZNGRkZGjdunGOZ3bp108yZM1WtWjXdfffd2rVrl1555RUNHjy4UMYIAACKnkINQH369NGZM2c0ZcoUnT59Wo0aNdK6descJ0YfP37caW/OlStXNGnSJP38888qVaqUunTpog8++ED+/v6OPnPnztXkyZP1t7/9TQkJCQoKCtKTTz6pKVOm3O7hAQCAIqpQ7wNUVHEfIAAAip9icR8gAACAwkIAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAlkMAAgAAllPoAWj+/PmqXr26PD091aJFC+3YsSPHvmlpaZoxY4Zq1qwpT09PhYaGat26dVn6nTx5Un/5y19Urlw5eXl5qWHDhtq5c2dBDgMAABQjhRqAli9frjFjxmjq1KmKjY1VaGioIiIilJCQkG3/SZMmaeHChZo7d67279+voUOHqmfPntq1a5ejz2+//aZWrVqpZMmS+vLLL7V//37Nnj1bZcqUuV3DAgAARZzNGGMK68VbtGihZs2aad68eZKkjIwMVa1aVSNHjtSECROy9A8KCtIzzzyj4cOHO9p69eolLy8vffjhh5KkCRMmaOvWrfrmm29uuq6kpCT5+fkpMTFRvr6+N70cAABw+7jy/V1oe4CuXr2q77//XuHh4f8rxs1N4eHh2rZtW7bzpKamytPT06nNy8tLW7ZscTxfs2aN7r33Xj366KOqWLGiGjdurLfffjvXWlJTU5WUlOT0AAAAd65CC0Bnz55Venq6AgICnNoDAgJ0+vTpbOeJiIjQK6+8okOHDikjI0MbNmzQypUrderUKUefn3/+WQsWLFDt2rW1fv16DRs2TKNGjdKSJUtyrCUqKkp+fn6OR9WqVfNnkAAAoEgq9JOgXfHaa6+pdu3aqlu3rjw8PDRixAgNGjRIbm7/G0ZGRoaaNGmi559/Xo0bN9aQIUP017/+VW+++WaOy504caISExMdjxMnTtyO4QAAgEJSaAGofPnycnd3V3x8vFN7fHy8AgMDs52nQoUKWr16tVJSUnTs2DEdOHBApUqVUo0aNRx9KlWqpPr16zvNV69ePR0/fjzHWux2u3x9fZ0eAADgzlVoAcjDw0NNmzZVdHS0oy0jI0PR0dEKCwvLdV5PT09VrlxZ165d02effabu3bs7prVq1UoHDx506v/f//5XwcHB+TsAAABQbJUozBcfM2aMIiMjde+996p58+aaM2eOUlJSNGjQIEnSgAEDVLlyZUVFRUmStm/frpMnT6pRo0Y6efKkpk2bpoyMDI0bN86xzKefflotW7bU888/r969e2vHjh1666239NZbbxXKGAEAQNFTqAGoT58+OnPmjKZMmaLTp0+rUaNGWrdunePE6OPHjzud33PlyhVNmjRJP//8s0qVKqUuXbrogw8+kL+/v6NPs2bNtGrVKk2cOFEzZsxQSEiI5syZo/79+9/u4QEAgCKqUO8DVFRxHyAAAIqfYnEfIAAAgMJCAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZDAAIAAJZTorALKIqMMZKkpKSkQq4EAADkVeb3dub3eG4IQNm4ePGiJKlq1aqFXAkAAHDVxYsX5efnl2sfm8lLTLKYjIwM/frrrypdurRsNlu+LjspKUlVq1bViRMn5Ovrm6/LLmoY653LSuNlrHcuK43XKmM1xujixYsKCgqSm1vuZ/mwBygbbm5uqlKlSoG+hq+v7x39JvwjxnrnstJ4Geudy0rjtcJYb7TnJxMnQQMAAMshAAEAAMshAN1mdrtdU6dOld1uL+xSChxjvXNZabyM9c5lpfFaaax5xUnQAADActgDBAAALIcABAAALIcABAAALIcABAAALIcAVADmz5+v6tWry9PTUy1atNCOHTty7f/pp5+qbt268vT0VMOGDfXFF1/cpkpvXlRUlJo1a6bSpUurYsWK6tGjhw4ePJjrPIsXL5bNZnN6eHp63qaKb960adOy1F23bt1c5ymO2zRT9erVs4zXZrNp+PDh2fYvTtt18+bN6tatm4KCgmSz2bR69Wqn6cYYTZkyRZUqVZKXl5fCw8N16NChGy7X1c/87ZLbeNPS0jR+/Hg1bNhQPj4+CgoK0oABA/Trr7/musyb+TzcDjfatgMHDsxSd6dOnW643KK4bW801uw+vzabTS+99FKOyyyq27UgEYDy2fLlyzVmzBhNnTpVsbGxCg0NVUREhBISErLt/5///Ef9+vXT448/rl27dqlHjx7q0aOH9u3bd5srd83XX3+t4cOH69tvv9WGDRuUlpamjh07KiUlJdf5fH19derUKcfj2LFjt6niW3P33Xc71b1ly5Yc+xbXbZrpu+++cxrrhg0bJEmPPvpojvMUl+2akpKi0NBQzZ8/P9vps2bN0uuvv64333xT27dvl4+PjyIiInTlypUcl+nqZ/52ym28ly5dUmxsrCZPnqzY2FitXLlSBw8e1J/+9KcbLteVz8PtcqNtK0mdOnVyqnvp0qW5LrOobtsbjfWPYzx16pTee+892Ww29erVK9flFsXtWqAM8lXz5s3N8OHDHc/T09NNUFCQiYqKyrZ/7969TdeuXZ3aWrRoYZ588skCrTO/JSQkGEnm66+/zrHPokWLjJ+f3+0rKp9MnTrVhIaG5rn/nbJNMz311FOmZs2aJiMjI9vpxXW7SjKrVq1yPM/IyDCBgYHmpZdecrRduHDB2O12s3Tp0hyX4+pnvrBcP97s7Nixw0gyx44dy7GPq5+HwpDdWCMjI0337t1dWk5x2LZ52a7du3c3HTp0yLVPcdiu+Y09QPno6tWr+v777xUeHu5oc3NzU3h4uLZt25btPNu2bXPqL0kRERE59i+qEhMTJUlly5bNtV9ycrKCg4NVtWpVde/eXT/++OPtKO+WHTp0SEFBQapRo4b69++v48eP59j3Ttmm0u/v6Q8//FCDBw/O9YeBi+t2/aO4uDidPn3aadv5+fmpRYsWOW67m/nMF2WJiYmy2Wzy9/fPtZ8rn4eiJCYmRhUrVlSdOnU0bNgwnTt3Lse+d8q2jY+P19q1a/X444/fsG9x3a43iwCUj86ePav09HQFBAQ4tQcEBOj06dPZznP69GmX+hdFGRkZGj16tFq1aqUGDRrk2K9OnTp677339K9//UsffvihMjIy1LJlS/3yyy+3sVrXtWjRQosXL9a6deu0YMECxcXFqXXr1rp48WK2/e+EbZpp9erVunDhggYOHJhjn+K6Xa+XuX1c2XY385kvqq5cuaLx48erX79+uf5Ypqufh6KiU6dOev/99xUdHa0XX3xRX3/9tTp37qz09PRs+98p23bJkiUqXbq0Hn744Vz7Fdfteiv4NXjcsuHDh2vfvn03PF4cFhamsLAwx/OWLVuqXr16WrhwoZ599tmCLvOmde7c2fHve+65Ry1atFBwcLA++eSTPP1VVZy9++676ty5s4KCgnLsU1y3K/4nLS1NvXv3ljFGCxYsyLVvcf089O3b1/Hvhg0b6p577lHNmjUVExOjBx54oBArK1jvvfee+vfvf8MLE4rrdr0V7AHKR+XLl5e7u7vi4+Od2uPj4xUYGJjtPIGBgS71L2pGjBihzz//XJs2bVKVKlVcmrdkyZJq3LixDh8+XEDVFQx/f3/dddddOdZd3LdppmPHjmnjxo164oknXJqvuG7XzO3jyra7mc98UZMZfo4dO6YNGzbkuvcnOzf6PBRVNWrUUPny5XOs+07Ytt98840OHjzo8mdYKr7b1RUEoHzk4eGhpk2bKjo62tGWkZGh6Ohop7+Q/ygsLMypvyRt2LAhx/5FhTFGI0aM0KpVq/TVV18pJCTE5WWkp6dr7969qlSpUgFUWHCSk5N15MiRHOsurtv0eosWLVLFihXVtWtXl+Yrrts1JCREgYGBTtsuKSlJ27dvz3Hb3cxnvijJDD+HDh3Sxo0bVa5cOZeXcaPPQ1H1yy+/6Ny5cznWXdy3rfT7HtymTZsqNDTU5XmL63Z1SWGfhX2nWbZsmbHb7Wbx4sVm//79ZsiQIcbf39+cPn3aGGPMY489ZiZMmODov3XrVlOiRAnz8ssvm59++slMnTrVlCxZ0uzdu7ewhpAnw4YNM35+fiYmJsacOnXK8bh06ZKjz/VjnT59ulm/fr05cuSI+f77703fvn2Np6en+fHHHwtjCHn297//3cTExJi4uDizdetWEx4ebsqXL28SEhKMMXfONv2j9PR0U61aNTN+/Pgs04rzdr148aLZtWuX2bVrl5FkXnnlFbNr1y7HVU8vvPCC8ff3N//617/MDz/8YLp3725CQkLM5cuXHcvo0KGDmTt3ruP5jT7zhSm38V69etX86U9/MlWqVDG7d+92+hynpqY6lnH9eG/0eSgsuY314sWLZuzYsWbbtm0mLi7ObNy40TRp0sTUrl3bXLlyxbGM4rJtb/Q+NsaYxMRE4+3tbRYsWJDtMorLdi1IBKACMHfuXFOtWjXj4eFhmjdvbr799lvHtLZt25rIyEin/p988om56667jIeHh7n77rvN2rVrb3PFrpOU7WPRokWOPtePdfTo0Y71EhAQYLp06WJiY2Nvf/Eu6tOnj6lUqZLx8PAwlStXNn369DGHDx92TL9TtukfrV+/3kgyBw8ezDKtOG/XTZs2Zfu+zRxPRkaGmTx5sgkICDB2u9088MADWdZBcHCwmTp1qlNbbp/5wpTbeOPi4nL8HG/atMmxjOvHe6PPQ2HJbayXLl0yHTt2NBUqVDAlS5Y0wcHB5q9//WuWIFNctu2N3sfGGLNw4ULj5eVlLly4kO0yist2LUg2Y4wp0F1MAAAARQznAAEAAMshAAEAAMshAAEAAMshAAEAAMshAAEAAMshAAEAAMshAAEAAMshAAGAJJvNptWrVxd2GQBuEwIQgEI3cOBA2Wy2LI9OnToVdmkA7lAlCrsAAJCkTp06adGiRU5tdru9kKoBcKdjDxCAIsFutyswMNDpUaZMGUm/H55asGCBOnfuLC8vL9WoUUMrVqxwmn/v3r3q0KGDvLy8VK5cOQ0ZMkTJyclOfd577z3dfffdstvtqlSpkkaMGOE0/ezZs+rZs6e8vb1Vu3ZtrVmzxmn6vn371LlzZ5UqVUoBAQF67LHHdPbsWcf0du3aadSoURo3bpzKli2rwMBATZs2LR/XEoD8QgACUCxMnjxZvXr10p49e9S/f3/17dtXP/30kyQpJSVFERERKlOmjL777jt9+umn2rhxo1PAWbBggYYPH64hQ4Zo7969WrNmjWrVquX0GtOnT1fv3r31ww8/qEuXLurfv7/Onz8vSbpw4YI6dOigxo0ba+fOnVq3bp3i4+PVu3dvp2UsWbJEPj4+2r59u2bNmqUZM2Zow4YNBbx2ALissH+NFQAiIyONu7u78fHxcXrMnDnTGGOMJDN06FCneVq0aGGGDRtmjDHmrbfeMmXKlDHJycmO6WvXrjVubm6OX/wOCgoyzzzzTI41SDKTJk1yPE9OTjaSzJdffmmMMebZZ581HTt2dJrnxIkTRpLjF+Pbtm1r7r//fqc+zZo1M+PHj3dpfQAoeJwDBKBIaN++vRYsWODUVrZsWce/w8LCnKaFhYVp9+7dkqSffvpJoaGh8vHxcUxv1aqVMjIydPDgQdlsNv3666964IEHcq3hnnvucfzbx8dHvr6+SkhIkCTt2bNHmzZtUqlSpbLMd+TIEd11111ZliFJlSpVciwDQNFBAAJQJPj4+GQ5JJVfvLy88tSvZMmSTs9tNpsyMjIkScnJyerWrZtefPHFLPNVqlQpT8sAUHRwDhCAYuHbb7/N8rxevXqSpHr16mnPnj1KSUlxTN+6davc3NxUp04dlS5dWtWrV1d0dPRNv36TJk30448/qnr16qpVq5bT4497ngAUDwQgAEVCamqqTp8+7fT44xVWn376qd577z3997//1dSpU7Vjxw7HSc79+/eXp6enIiMjtW/fPm3atEkjR47UY489poCAAEnStGnTNHv2bL3++us6dOiQYmNjNXfu3DzXN3z4cJ0/f179+vXTd999pyNHjmj9+vUaNGiQ0tPT83dlAChwHAIDUCSsW7fO6VCSJNWpU0cHDhyQ9PsVWsuWLdPf/vY3VapUSUuXLlX9+vUlSd7e3lq/fr2eeuopNWvWTN7e3urVq5deeeUVx7IiIyN15coVvfrqqxo7dqzKly+vRx55JM/1BQUFaevWrRo/frw6duyo1NRUBQcHq1OnTnJz429JoLixGWNMYRcBALmx2WxatWqVevToUdilALhD8GcLAACwHAIQAACwHM4BAlDkcaQeQH5jDxAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALAcAhAAALCc/wfHa1Z8CyjEnQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "result = model.evaluate(X_train, y_train)\n",
        "loss, accuracy = result  # Hier wird das Tupel aufgelöst\n",
        "print(f\"Test Loss: {loss}, Test Accuracy: {accuracy}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RPe2hcD_Ymel",
        "outputId": "4b8d88d6-a4e2-4adf-a5d0-b02c54217a1a"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 12s/step - accuracy: 1.0000 - loss: 5.1313e-07\n",
            "Test Loss: 5.131338980390865e-07, Test Accuracy: 1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Vorhersage machen. Das sind die Outputs für die Vorhersage von X_test.\n",
        "binäre_vorhersage = (model.predict(X_test) > threshold).astype(int)\n",
        "class_labels = np.where(binäre_vorhersage == 1, \"Blutgerinnung\", \"Keine Blutgerinnung\")\n",
        "print(class_labels)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fFm6tpaeYq32",
        "outputId": "18ef54f8-3b95-4aaf-e399-155a8dd200dd"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 7s/step\n",
            "[['Keine Blutgerinnung']]\n"
          ]
        }
      ]
    }
  ]
}