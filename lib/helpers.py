

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def getprogressplot(trainloss, validloss, acc, nullac, novacc, pulsatoracc, transitacc, EPOCHS, e):
  fig = go.Figure()


  fig.add_trace(go.Scatter(x=list(range(EPOCHS)), y=trainloss, mode='lines', name='Training Loss', line=dict(color='blue')))

  fig.add_trace(go.Scatter(x=list(range(EPOCHS)), y=validloss, mode='lines', name='Validation Loss', line=dict(color='orange')))

  fig.add_trace(go.Scatter(x=list(range(EPOCHS)), y=nullac, mode='lines', name='Null Accuracy', line=dict(color='gray')))

  fig.add_trace(go.Scatter(x=list(range(EPOCHS)), y=novacc, mode='lines', name='Nova Accuracy', line=dict(color='yellow')))

  fig.add_trace(go.Scatter(x=list(range(EPOCHS)), y=pulsatoracc, mode='lines', name='Pulsator Accuracy', line=dict(color='green')))

  fig.add_trace(go.Scatter(x=list(range(EPOCHS)), y=transitacc, mode='lines', name='Transit Accuracy', line=dict(color='purple')))

  fig.add_trace(go.Scatter(x=list(range(EPOCHS)), y=acc, mode='lines', name='Transit Accuracy', line=dict(color='red')))



  fig.update_yaxes(range=[0, 2])

  fig.update_xaxes(range=[0, EPOCHS])

  fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='top'))

  fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
  fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

  fig.update_xaxes(tickfont=dict(size=12))
  fig.update_yaxes(tickfont=dict(size=12))

  fig.update_layout(title='Training and Validation Loss Over Epochs: {}/{}'.format(e, EPOCHS),
                    xaxis_title='Epochs',
                    yaxis_title='Loss')

  return fig

def plot_from_tensor(data):
  fig = go.Figure()

  w1 = data[:, 0].numpy()
  w2 = data[:, 2].numpy()


  dt = data[:, -2].numpy()
  day = data[:, -1].numpy()


  fig.add_trace(go.Scatter(x=day, y=w1, marker=dict(size=5, opacity=0.7), name="w1mpro z-scored", mode='markers'))
  fig.add_trace(go.Scatter(x=day, y=w2, marker=dict(size=5, opacity=0.7), name="w2mpro z-scored", mode='markers'))

  fig.layout.width = 800
  fig.layout.height = 0.65 * fig.layout.width


  return fig