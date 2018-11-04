# coding=utf8
history = model.fit(x, y, batch_size=32, validation_split=0.2)
epochs = 30
train_acc, val_acc = history.history['acc'], history.history['val_acc']
train_loss, val_loss = history.history['loss'], history.history['val_loss']

plt.plot(range(1, epochs+1), train_acc, 'r-', range(1, epochs+1), val_acc, 'bo')
plt.legend(['train accuracy', 'val accuracy'])
plt.show()

plt.plot(range(1, epochs+1), train_loss, 'r-', range(1, epochs+1), val_loss, 'bo')
plt.legend(['train loss', 'val loss'])
plt.show()
