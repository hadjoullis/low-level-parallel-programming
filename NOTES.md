### alternatives:
- the following for loop resulted in considerably performance than the foreach
version
```
		for (int i = 0; i < this->agents.size(); i++) {
			agents[i]->computeNextDesiredPosition();
			int x = agents[i]->getDesiredX();
			int y = agents[i]->getDesiredY();
			agents[i]->setX(x);
			agents[i]->setY(y);
		}
```
