def checkConvergence(scores:list[list[float, float, float]], patience:int, threshold:float=0.01)->bool:
    if len(scores) < patience:
        return False

    recent_scores = scores[-patience]
    for node in range(len(scores[-1])):
      diff = recent_scores[node] - scores[-1][node]
      if diff > threshold:
          return False
      if diff < -threshold:
          return False

    return True