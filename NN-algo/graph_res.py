import matplotlib.pyplot as plt
import os


def plot_results(training_steps, training_losses, training_accuracy,
                 evaluation_steps, evaluation_losses, evaluation_accuracy,
                 evaluation_losses_noncovid, evaluation_accuracy_noncovid,
                 xmax, ymax, xmax2, ymax2):

    fig, graphs = plt.subplots(3, 2)
    fig.canvas.set_window_title('Results')
    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    graphs[0, 0].set_title("Training Loss")
    graphs[0, 0].set(xlabel="batch #", ylabel="loss")
    graphs[0, 1].set(xlabel="batch #", ylabel="accuracy")
    graphs[0, 1].set_title("Training Accuracy")

    graphs[1, 0].set_title("Validation Loss (Covid Tweets)")
    graphs[1, 0].set(xlabel="batch #", ylabel="loss")
    graphs[1, 1].set(xlabel="batch #", ylabel="accuracy")
    graphs[1, 1].set_title("Validation Accuracy (Covid Tweets)")

    graphs[2, 0].set_title("Validation Loss (Non-Covid Tweets)")
    graphs[2, 0].set(xlabel="batch #", ylabel="loss")
    graphs[2, 1].set(xlabel="batch #", ylabel="accuracy")
    graphs[2, 1].set_title("Validation Accuracy (Non-Covid Tweets)")

    graphs[0, 0].plot(training_steps, training_losses)
    graphs[0, 1].plot(training_steps, training_accuracy, 'tab:orange')
    graphs[1, 0].plot(evaluation_steps, evaluation_losses, 'tab:green')
    graphs[1, 1].plot(evaluation_steps, evaluation_accuracy, 'tab:red')
    graphs[2, 0].plot(evaluation_steps, evaluation_losses_noncovid,
                      'tab:green')
    graphs[2, 1].plot(evaluation_steps, evaluation_accuracy_noncovid,
                      'tab:pink')

    graphs[1, 1].annotate("max: ({}, {:.3f})".format(xmax, ymax),
                          xy=(xmax, ymax),
                          xytext=(xmax - 5, ymax - 0.05))
    graphs[2, 1].annotate("max: ({}, {:.3f})".format(xmax2, ymax2),
                          xy=(xmax2, ymax2),
                          xytext=(xmax2 - 5, ymax2 - 0.04))

    plt.savefig('summary.png')
    #plt.show()
