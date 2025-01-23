#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 100;       // Tamanho do tabuleiro
    vector<int> board(N, 0); // Tabuleiro com N posições inicializadas como 0
    const int goal = N - 1;  // Posição final (meta)
    srand(time(0) + rank);   // Semente para números aleatórios, diferenciada por rank

    int position = 0; // Posição inicial de cada processo
    bool winner = false;

    if (rank == 0)
    {
        cout << "Iniciando o jogo de corrida com " << size << " jogadores!" << endl;
        cout << "O primeiro jogador a alcançar a posição " << goal << " vence!" << endl;
    }

    while (!winner)
    {
        // Cada processo lança o dado
        int dice_roll = (rand() % 6) + 1;
        position += dice_roll;

        // Evita ultrapassar o limite do tabuleiro
        if (position >= goal)
        {
            position = goal;
            winner = true;
        }

        // Atualiza o tabuleiro para o processo atual
        board[position] = rank;

        // Sincroniza o estado do jogo
        MPI_Allreduce(MPI_IN_PLACE, &winner, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        if (winner)
        {
            // Determina qual processo venceu
            if (position == goal)
            {
                cout << "Jogador " << rank << " venceu o jogo!" << endl;
            }
        }

        // Sincroniza para a próxima rodada
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}