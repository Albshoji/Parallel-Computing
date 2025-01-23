#include <iostream>
#include <cmath>
#include <string>
#include <omp.h>
using namespace std;

// Função para criar o histograma a partir da imagem
void create_histogram(int *hist, int **img, int num_rows, int num_cols)
{
    int smallMatrix[3][3];
    int decimal;

#pragma omp parallel for collapse(2) private(smallMatrix, decimal)
    for (int i = 1; i <= num_rows; i++)
    {
        for (int j = 1; j <= num_cols; j++)
        {
            smallMatrix[0][0] = (img[i][j] > img[i - 1][j - 1]) ? 1 : 0;
            smallMatrix[0][1] = (img[i][j] > img[i - 1][j]) ? 1 : 0;
            smallMatrix[0][2] = (img[i][j] > img[i - 1][j + 1]) ? 1 : 0;
            smallMatrix[1][0] = (img[i][j] > img[i][j - 1]) ? 1 : 0;
            smallMatrix[1][2] = (img[i][j] > img[i][j + 1]) ? 1 : 0;
            smallMatrix[2][0] = (img[i][j] > img[i + 1][j - 1]) ? 1 : 0;
            smallMatrix[2][1] = (img[i][j] > img[i + 1][j]) ? 1 : 0;
            smallMatrix[2][2] = (img[i][j] > img[i + 1][j + 1]) ? 1 : 0;

            decimal = smallMatrix[0][0] * 128 + smallMatrix[0][1] * 64 + smallMatrix[0][2] * 32 +
                      smallMatrix[1][2] * 16 + smallMatrix[2][2] * 8 + smallMatrix[2][1] * 4 +
                      smallMatrix[2][0] * 2 + smallMatrix[1][0];

#pragma omp atomic
            hist[decimal]++;
        }
    }
}

// Função para calcular a distância entre dois histogramas
double distance(int *a, int *b, int size)
{
    double dist = 0.0;

#pragma omp parallel for reduction(+ : dist)
    for (int i = 0; i < size; i++)
    {
        if (a[i] + b[i] != 0)
        {
            dist += 0.5 * pow(a[i] - b[i], 2) / (a[i] + b[i]);
        }
    }

    return dist;
}

// Função para encontrar o índice mais próximo no conjunto de treinamento
int find_closest(int ***training_set, int num_persons, int num_training, int size, int *test_image)
{
    double min_distance = 1e9;
    int closest_person = 0;

#pragma omp parallel for
    for (int i = 0; i < num_persons; i++)
    {
        for (int j = 0; j < num_training; j++)
        {
            double dist = distance(training_set[i][j], test_image, size);
#pragma omp critical
            {
                if (dist < min_distance)
                {
                    min_distance = dist;
                    closest_person = i;
                }
            }
        }
    }

    return closest_person + 1;
}

// Função principal
int main(int argc, char *argv[])
{
    int num_persons = 18;
    int num_training = 10; // Número de imagens de treinamento
    int num_rows = 200, num_cols = 180, hist_size = 256;

    int ***training_set = new int **[num_persons];
#pragma omp parallel for
    for (int i = 0; i < num_persons; i++)
    {
        training_set[i] = new int *[num_training];
        for (int j = 0; j < num_training; j++)
        {
            training_set[i][j] = new int[hist_size]();
        }
    }

    // Inicializa matrizes e carrega imagens de teste (simulado aqui)
    for (int i = 0; i < num_persons; i++)
    {
        for (int j = 0; j < num_training; j++)
        {
            // Simulação: preenche matrizes com valores aleatórios
            for (int k = 0; k < hist_size; k++)
            {
                training_set[i][j][k] = rand() % 10;
            }
        }
    }

    int correct_predictions = 0, total_tests = 0;

    // Testa as imagens
    for (int i = 0; i < num_persons; i++)
    {
        for (int j = num_training; j < num_training + 5; j++)
        {                                         // Simulação de testes
            int *test_image = training_set[i][0]; // Usa a primeira imagem como teste
            int predicted = find_closest(training_set, num_persons, num_training, hist_size, test_image);

            total_tests++;
            if (predicted == i + 1)
                correct_predictions++;

            cout << "Test Image " << i + 1 << "." << j + 1 << ": Predicted = " << predicted
                 << ", Actual = " << i + 1 << endl;
        }
    }

    cout << "Accuracy: " << (double)correct_predictions / total_tests * 100 << "%" << endl;

#pragma omp parallel for
    for (int i = 0; i < num_persons; i++)
    {
        for (int j = 0; j < num_training; j++)
        {
            delete[] training_set[i][j];
        }
        delete[] training_set[i];
    }
    delete[] training_set;

    return 0;
}
